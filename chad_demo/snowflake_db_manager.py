import logging
import os
import string
import random
import time
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.types import NVARCHAR, DATETIME
from sqlalchemy import inspect, Table, MetaData
from typing import Optional, Tuple, Sequence
from snowflake.connector import ProgrammingError
from snowflake.connector.pandas_tools import chunk_helper, T
from tempfile import TemporaryDirectory
import snowflake.connector
import parameters as p
import re    

class SNOWFLAKE_Config():
    '''
    SNOWFLAKE Config class
    '''
    def __init__(self):
        self.host = p.sf_host
        self.username = p.sf_username
        self.password = p.sf_password
        self.account_name = p.sf_account_name
        self.role = p.sf_role
        self.warehouse = p.sf_warehouse
        self.db_name = p.sf_dbname

    def get_connection_string(self, schema: str):
        self.schema = schema
        connection_str = self.host + '://' + self.username + ':' + \
            self.password + '@' + \
            self.account_name + '/' + \
            self.db_name + '/' + \
            schema + '?warehouse=' + \
            self.warehouse + '&role=' + \
            self.role
        return connection_str

class Snowflake_DBManager:
    '''
    SF DB manager
    '''
    def __init__(self, database: str):
        self.db_name = database
        self.config = SNOWFLAKE_Config()
        self.engine = sa.create_engine(
            self.config.get_connection_string(database)
        )
        self.data_types = {
            'float16': 'FLOAT',
            'float32': 'FLOAT',
            'float64': 'FLOAT',
            'object': 'VARCHAR(16777216)',
            'int32': 'INTEGER',
            'int64': 'INTEGER',
            'bool': 'BOOLEAN',
            'datetime64[ns]': 'DATETIME',
            'datetime64[ns, UTC]': 'DATETIME',
            'timedelta[ms]': 'DATETIME',
            'datetime.datetime': 'DATETIME'
        }
    
    @staticmethod
    def title_case_col_names(df_columns, separator='_'):
        """
        All words separated by _ will be uppercase
        :param df_columns: list
            Names that should be converted into uppercase
        :param separator: str
            Type of separator that is used to concatenate results
        :return:
            df_columns with all elements separated by _ will be uppercase
        """
        title_case_name_list = {}
        for raw_col in df_columns:
            title_case_col_name = separator.join([sub_col.capitalize() for sub_col in re.split("[, \-_]+", raw_col)])
            title_case_name_list.update({raw_col: title_case_col_name})
        return title_case_name_list
    
    def read_sample_data(self):
        sql_query = f'SELECT * FROM {self.config.db_name}.{self.config.schema}.SHAP_1'
        shap_df = pd.read_sql_query(sql_query, self.engine)
        return shap_df
    
    def check_table_name(self,
                         table_name: str
                         ) -> Tuple[str, str]:
        table_name = table_name.lower()
        inspector = inspect(self.engine)
        tableNameList = inspector.get_table_names(schema=self.config.schema)
        if table_name not in tableNameList:
            table_name = None
        return self.config.schema, table_name

    def write_db(self,
                  df: pd.DataFrame,
                  table_name: str,
                  asset_name: str,
                  if_exists: Optional[str] = 'append',
                  chunksize: Optional[int] = 10000,
                  data_type_dict: Optional[dict] = {}):

        # make sure column names are all string, or parquet will fail
        df.columns = df.columns.astype(str)
        df.rename(columns=self.title_case_col_names(df.columns), inplace=True)

        for col in df:
            # Added UTC TimeZone as requested per snowflake
            # issue https://github.com/snowflakedb/snowflake-connector-python/issues/319
            if df[col].dtype == 'datetime64[ns]':
                df[col] = df[col].dt.tz_localize('UTC')
                data_type_dict[col] = DATETIME
            elif col not in data_type_dict:
                if df[col].dtype == str or df[col].dtype == object:
                    data_type_dict[col] = NVARCHAR
        
        # check if table exists, if yes then remove conflict rows if needed
        _, tb = self.check_table_name(table_name)
        if tb is not None and if_exists == 'append':
            metadata = MetaData(self.engine)
            db_table = Table(table_name, metadata, autoload=True)
            db_table.delete().where(db_table.c.asset == asset_name).execute()
        else: # if not then create table
            columns_structure = [str(col) + ' ' + self.data_types[str(df[col].dtypes)] for col in df.columns]
            raw_columns = (', '.join('{0}'.format(w) for w in columns_structure))
            query = "CREATE OR REPLACE TABLE" " {db_name}.{schema}.{table_name}(".format(
                table_name=table_name,
                db_name=self.config.db_name,
                schema=self.config.schema) + raw_columns + ")"

            con = self.__get_connection()
            cur = con.cursor()
            try:
                cur.execute(query)
                con.commit()
            except Exception as e:
                con.rollback()
                logging.error('ERROR %s', e)
            finally:
                con.close()

        if len(df) > 0:
            con = self.__get_connection()
            try:
                self.write_pandas(conn=con, df=df, table_name=table_name, quote_identifiers=False)
                con.commit()
            except Exception as e:
                con.rollback()
                logging.error('ERROR %s', e)                
            finally:
                con.close()

    def __get_connection(self):
        tries = 0
        while True:
            tries += 1
            if tries > 5:
                raise Exception("Failed connect to database")
            try:
                connection = snowflake.connector.connect(
                            account=self.config.account_name,
                            user=self.config.username,
                            password=self.config.password,
                            database=self.config.db_name,
                            schema=self.db_name ,
                            warehouse=self.config.warehouse,
                            role=self.config.role,
                            autocommit=False
                    )

                logging.info('Connected to DB')
                return connection
            except Exception as e:
                logging.error("%s", e)
                logging.info("LOG: could not connect to db, retrying...")
                time.sleep(10)
                
    def write_pandas(self, conn: 'SnowflakeConnection',
                     df: pd.DataFrame,
                     table_name: str,
                     database: Optional[str] = None,
                     schema: Optional[str] = None,
                     chunk_size: Optional[int] = None,
                     compression: str = 'gzip',
                     on_error: str = 'abort_statement',
                     parallel: int = 4,
                     quote_identifiers: bool = False
                     ) -> Tuple[bool, int, int,
                                Sequence[Tuple[str, str, int, int, int, int, Optional[str], Optional[int],
                                               Optional[int], Optional[str]]]]:
        """Allows users to most efficiently write back a pandas DataFrame to Snowflake.

        It works by dumping the DataFrame into Parquet files, uploading them and finally copying their data into the table.

        Returns whether all files were ingested correctly, number of chunks uploaded, and number of rows ingested
        with all of the COPY INTO command's output for debugging purposes.

            Example usage:
                import pandas
                from snowflake.connector.pandas_tools import write_pandas

                df = pandas.DataFrame([('Mark', 10), ('Luke', 20)], columns=['name', 'balance'])
                success, nchunks, nrows, _ = write_pandas(cnx, df, 'customers')

        Args:
            conn: Connection to be used to communicate with Snowflake.
            df: Dataframe we'd like to write back.
            table_name: Table name where we want to insert into.
            database: Database schema and table is in, if not provided the default one will be used (Default value = None).
            schema: Schema table is in, if not provided the default one will be used (Default value = None).
            chunk_size: Number of elements to be inserted once, if not provided all elements will be dumped once
                (Default value = None).
            compression: The compression used on the Parquet files, can only be gzip, or snappy. Gzip gives supposedly a
                better compression, while snappy is faster. Use whichever is more appropriate (Default value = 'gzip').
            on_error: Action to take when COPY INTO statements fail, default follows documentation at:
                https://docs.snowflake.com/en/sql-reference/sql/copy-into-table.html#copy-options-copyoptions
                (Default value = 'abort_statement').
            parallel: Number of threads to be used when uploading chunks, default follows documentation at:
                https://docs.snowflake.com/en/sql-reference/sql/put.html#optional-parameters (Default value = 4).
            quote_identifiers: By default, identifiers, specifically database, schema, table and column names
                (from df.columns) will be quoted. If set to False, identifiers are passed on to Snowflake without quoting.
                I.e. identifiers will be coerced to uppercase by Snowflake.  (Default value = True)

        Returns:
            Returns the COPY INTO command's results to verify ingestion in the form of a tuple of whether all chunks were
            ingested correctly, # of chunks, # of ingested rows, and ingest's output.
        """
        if database is not None and schema is None:
            raise ProgrammingError("Schema has to be provided to write_pandas when a database is provided")
        # This dictionary maps the compression algorithm to Snowflake put copy into command type
        # https://docs.snowflake.com/en/sql-reference/sql/copy-into-table.html#type-parquet
        compression_map = {
            'gzip': 'auto',
            'snappy': 'snappy'
        }
        if compression not in compression_map.keys():
            raise ProgrammingError("Invalid compression '{}', only acceptable values are: {}".format(
                compression,
                compression_map.keys()
            ))
        if quote_identifiers:
            location = ((database + '.' if database else '') +
                        (schema + '.' if schema else '') +
                        (table_name))
        else:
            location = ((database + '.' if database else '') +
                        (schema + '.' if schema else '') +
                        (table_name))
        if chunk_size is None:
            chunk_size = len(df)
        cursor = conn.cursor()
        stage_name = None  # Forward declaration
        while True:
            try:
                stage_name = ''.join(random.choice(string.ascii_lowercase) for _ in range(5))
                create_stage_sql = ('create temporary stage /* Python:snowflake.connector.pandas_tools.write_pandas() */ '
                                    '"{stage_name}"').format(stage_name=stage_name)
                # logging.debug("creating stage with '{}'".format(create_stage_sql))
                cursor.execute(create_stage_sql, _is_internal=True).fetchall()
                break
            except ProgrammingError as pe:
                if pe.msg.endswith('already exists.'):
                    continue
                raise

        with TemporaryDirectory() as tmp_folder:
            for i, chunk in chunk_helper(df, chunk_size):
                chunk_path = os.path.join(tmp_folder, 'file{}.txt'.format(i))
                # Dump chunk into parquet file
                chunk.to_parquet(chunk_path, compression=compression, engine='fastparquet')
                # Upload parquet file
                upload_sql = ('PUT /* Python:snowflake.connector.pandas_tools.write_pandas() */ '
                            '\'file://{path}\' @"{stage_name}" PARALLEL={parallel}').format(
                    path=chunk_path.replace('\\', '\\\\').replace('\'', '\\\''),
                    stage_name=stage_name,
                    parallel=parallel
                )
                # logging.debug("uploading files with '{}'".format(upload_sql))
                cursor.execute(upload_sql, _is_internal=True)
                # Remove chunk file
                os.remove(chunk_path)
        if quote_identifiers:
            columns = '"' + '","'.join(list(df.columns)) + '"'
            df.columns = ['"' +str(col) + '"' for col in df.columns]
        else:
            columns = ','.join(list(df.columns))

        # in Snowflake, all parquet data is stored in a single column, $1, so we must select columns explicitly
        # see (https://docs.snowflake.com/en/user-guide/script-data-load-transform-parquet.html)
        # parquet_columns = '$1:' + ',$1:'.join(df.columns)
        parquet_columns = 'nullif($1:' + ',\'nan\'),nullif($1:'.join(df.columns)+',\'nan\')'
        copy_into_sql = ('COPY INTO {location} /* Python:snowflake.connector.pandas_tools.write_pandas() */ '
                        '({columns}) '
                        'FROM (SELECT {parquet_columns} FROM @"{stage_name}") '
                        'FILE_FORMAT=(TYPE=PARQUET COMPRESSION={compression} ) '
                        'PURGE=TRUE ON_ERROR={on_error}').format(
            location=location,
            columns=columns,
            parquet_columns=parquet_columns,
            stage_name=stage_name,
            compression=compression_map[compression],
            on_error=on_error
        )
        # logging.debug("copying into with '{}'".format(copy_into_sql))
        copy_results = cursor.execute(copy_into_sql, _is_internal=True).fetchall()
        cursor.close()
        return (all(e[1] == 'LOADED' for e in copy_results),
                len(copy_results),
                sum(e[3] for e in copy_results),
                copy_results)