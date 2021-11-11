# mappings in the cmt_renamed scenario
# should be relavant for all cmt scenarios

cmt_renamed_mapping = {
    "Author": "select id from authors;",
    "Person": "select id from persons;",
    "Co-author": "select id from co_authors;",
    "Conference": "select id from conferences;",
    "Reviewer": "select id from reviewers;",
    "Document": "select id from documents;",
    "PaperFullVersion": "select id from paper_full_versions;",
    "PaperAbstract": "select id from paper_abstracts;",
    "Review": "select id from reviews;",
    'ProgramCommittee': 'select id from program_committees;',
    'ProgramCommitteeMember': 'select id from pc_members;',
    'ProgramCommitteeChair': 'select id from program_committee_chairs;',    
    'name': 'select id, name from persons union (select id, name from conferences);',
    'email': 'select id, email from persons;',
    'date': 'select id, date from conferences;',
    'label': 'select id, label from program_committees;',
    'paperID': 'select id, paper_id from papers;',
    'title': 'select id, title from papers;',
    'comment': 'select id, comment from reviews;',
    'siteURL': 'select id, site_url from conferences;',
    'hasAuthor': 'select id, author from papers;',
    'co-writePaper': 'select cid, pid from co_author_paper;',
    'hasBeenAssigned': 'select rid, pid from paper_reviewer;',
    'writeReview': 'select written, id from reviews;',
    'hasProgramCommitteeMember': 'select program_committee, program_committee_member from program_committee_members join conf_members on conf_members.id=program_committee_members.program_committee_member;',
    'hasConferenceMember': 'select conference, conference_member from conference_members join conf_members on conf_members.id=conference_members.conference_member;',
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
    'name': 'select "ID", "name" from "Person" union (select "ID", "name" from "Conference");', 
    'email': 'select "ID, "email" from "Person";',
    'date': 'select "ID", "date" from "Conference";',    
    'label': 'select "ID", "label" from "ProgramCommittee";',
    'paperID': 'select "ID", "paperID" from "Paper";',
    'title': 'select "ID", "title" from "Paper";',
    'comment': 'select "ID", "comment" from "Review";',
    'siteURL': 'select "ID", "siteURL" from "Conference";',
    'hasAuthor': 'select "ID", "hasAuthor" from "Paper";',
    'co-writePaper': 'select "Co-author", "Paper" from "co_writePaper";',
    'hasBeenAssigned': 'select "Reviewer", "Paper" from "assignedTo";',    
    'writeReview': 'select "writtenBy", "ID" from "Review";',
    'hasProgramCommitteeMember': 'select "ProgramCommittee", "ProgramCommitteeMember" from "hasProgramCommitteeMember"  join "ProgramCommitteeMember" on "hasProgramCommitteeMember"."ProgramCommitteeMember"="ProgramCommitteeMember"."ID";',
    'hasConferenceMember': 'select "Conference", "ConferenceMember" from "hasConferenceMember";',
}

cmt_mapping = cmt_renamed_mapping
cmt_schema = "cmt_renamed"
    
