# mappings in the cmt_renamed scenario
# should be relavant for all cmt scenarios

cmt_schema = "cmt_renamed"

cmt_mapping = {
    "Person": "select id from persons;",
    "Author": "select id from authors;",
    "Co-author": "select id from co_authors;",
    "Conference": "select id from conferences;",
    "Reviewer": "select id from reviewers;",
    "Document": "select id from documents;",
    "PaperFullVersion": "select id from paper_full_versions;",
    "PaperAbstract": "select id from paper_abstracts;",
    "Review": "select id from reviews;",    
}
