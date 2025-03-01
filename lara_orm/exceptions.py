class ORMError(Exception):
    pass

class NotFoundError(ORMError):
    pass

class RelationshipError(ORMError):
    pass

class EventCancelledError(ORMError):
    pass