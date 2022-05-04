from marshmallow import fields, Schema
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class IncomingMessage:
    play: str
    drilling_days: int
    mod_factor: float

@dataclass
class OutgoingMessage:
    code: int
    messsage: str
    results: Dict

class IncomingSchema(Schema):
    '''Incoming message schema'''
    play = fields.String(attribute="play")
    drilling_days = fields.Integer(attribute="drilling_days")
    mod_factor = fields.Float(attribute="mod_factor")

class OutgoingSchema(Schema):
    '''Outgoing message schema'''
    code = fields.Integer(attribute="code")
    messsage = fields.String(attribute="messsage")
    results = fields.Dict(attribute="results")