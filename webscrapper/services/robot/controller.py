import traceback
from flask import request, abort
from flask_accepts import accepts, responds
from flask_restx import Namespace, Resource

from .service import RobotService
from .schema import OutgoingMessage, IncomingSchema, OutgoingSchema

api = Namespace('RobotService', description='Powertrain Automated UI Service')
service = RobotService()

def build_outgoing_message(code, message, prediction_result):
    return OutgoingSchema().dump(OutgoingMessage(code, message, prediction_result))

@api.route('/drilling_days', methods=['POST'])
class DrillingDaysResource(Resource):
    @api.doc(responses={ 200: 'Success', 413: 'Request Entity Too Large', 500: 'Internal Server Error' })
    @accepts(schema=IncomingSchema, api=api)
    @responds(schema=OutgoingSchema)
    def post(self) -> OutgoingSchema:
        '''
        Set drilling days and return new costs
        '''
        try:
            input_data = request.parsed_obj
            drilling_days = input_data['drilling_days']
            result = service.set_drilling_days(drilling_days)
            return build_outgoing_message(200, 'Success', result)
        except Exception as e:
            abort(500, f'Error: {str(e)}\r\n{traceback.format_exc()}')

@api.route('/mod_factor', methods=['POST'])
class ModFactorResource(Resource):
    @api.doc(responses={ 200: 'Success', 413: 'Request Entity Too Large', 500: 'Internal Server Error' })
    @accepts(schema=IncomingSchema, api=api)
    @responds(schema=OutgoingSchema)
    def post(self) -> OutgoingSchema:
        '''
        Set modification factor and return new costs
        '''
        try:
            input_data = request.parsed_obj
            mod_factor = input_data['mod_factor']
            result = service.set_mod_factor(mod_factor)
            return build_outgoing_message(200, 'Success', result)
        except Exception as e:
            abort(500, f'Error: {str(e)}\r\n{traceback.format_exc()}')