"""Classical geometric registration stage of CourtAlign-2S."""

__all__ = ["CourtAlign2STennisFullCourtRegistration"]


def __getattr__(name):
    if name == "CourtAlign2STennisFullCourtRegistration":
        from courtalign_2s.registration.tennis_fullcourt import CourtAlign2STennisFullCourtRegistration

        return CourtAlign2STennisFullCourtRegistration
    raise AttributeError(name)
