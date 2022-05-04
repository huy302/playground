import parameters as p

def register_routes(api, app, root="api"):
    from .controller import api as robot_api
    api.add_namespace(robot_api, path=f"/{root}/{p.VERSION}/robot")