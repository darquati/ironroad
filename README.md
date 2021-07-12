# IronRoad

IronRoad will be a suite of Python tools to process public transport timetable data and generate insights about the quality of service.

It is still under development but broadly consists of:
- timetables
- locations
- routes
- quality

## Timetables
This submodule allows the loading, processing, enrichment and saving of PT timetables, currently just in Network Rail's CIF format and the TransXChange-based format (with only Transport for London's releases currently implemented).

## Locations
This allows the conversion of various location coding systems including NR's TIPLOCs & CRS codes, DfT's NaPTaN codes, and TfL's various three- and four-letter code formats.

## Routes
This implements the Microsoft RAPTOR algorithm in Python to allow pathfinding through the timetables.

## Quality
Not yet ready - allows measurement of the quality of service (either one-dimensionally at a location, between a pair or set of locations, or generally for a line, network or area.

## Reference data
Not yet ready but allows all of the above to carry out lookups and conversions.
