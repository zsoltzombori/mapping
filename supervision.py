# mappings in the cmt_renamed scenario
# should be relavant for all cmt scenarios

cmt_renamed_mapping = {
    # "Author": "select id from authors;",
    # "Person": "select id from persons;",
    # "Co-author": "select id from co_authors;",
    # "Conference": "select id from conferences;",
    # "Reviewer": "select id from reviewers;",
    # "Document": "select id from documents;",
    # "PaperFullVersion": "select id from paper_full_versions;",
    # "PaperAbstract": "select id from paper_abstracts;",
    # "Review": "select id from reviews;",    



    'name': 'select id, name from persons;', # TODO binary predicates may have several variants
}

cmt_structured_mapping = {
    "Author": 'select "ID" from "Person" where "is_Author"=true;',
    "Person": 'select "ID" from "Person";',
    "Co-author": 'select "ID" from "Person" where "is_Co-author"=true;',
    'Conference': 'select "ID" from "Conference;',
    'Reviewer': 'select "ID" from "Person" where "is_Reviewer"=true;',
    'Document': 'select "ID" from "Document";',
    'PaperFullVersion': 'select "ID" from "Paper" where "TYPE"=1;',
    'PaperAbstract': 'select "ID" from "Paper" where "TYPE"=2;',
    'Review': 'select "ID" from "Review";',
    'ProgramCommittee': 'select "ID" from "ProgramCommittee";',
    'ProgramCommitteeMember': 'select "ID" from "ProgramCommitteeMember";',
    'ProgramCommitteeChair': 'select "ID" from "ProgramCommitteeChair";',
    'name': 'select "ID", "name" from "Person";', # TODO binary predicates may have several variants
    
}

cmt_mapping = cmt_renamed_mapping
cmt_schema = "cmt_renamed"
    
