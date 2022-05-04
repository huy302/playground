def register_routes(api, app, root="api"):
    from services.robot import register_routes as attach_robot
    
    attach_robot(api, app)