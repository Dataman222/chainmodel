class Config:
    # Input Columns
    TICKET_SUMMARY = 'Ticket Summary'
    INTERACTION_CONTENT = 'Interaction content'

    # Type Columns to test

    # commenting out original multiclass code
    # TYPE_COLS = ['y2', 'y3', 'y4']
    # CLASS_COL = 'y2'
    # GROUPED = 'y1'

    # new multilabel code placeholder, to be reviewed
    TYPE_COLS = ['y2', 'y3', 'y4']
    CLASS_COL = ['y2', 'y3', 'y4']
    GROUPED = 'y1'
    GROUPED2 = 'y2'
    GROUPED3 = 'y3'