import rdflib
import os

import mappingProblem
import query
import util

schema = "cmt_renamed"
ontology = "RODI/data/cmt_renamed/ontology.ttl"
query_dir = "RODI/data/cmt_renamed/queries"

mappings = {
    ":Person": ("persons", "id"),
    ":Conference": ("conferences", "id"),
    ":siteURL": ("conferences", "id", "site_url"),
    ":Review": ("reviews", "id"),
    ":Reviewer": ("reviewers", "id"),
    ":writeReview": ("reviews", "id", "written"),
    ":hasConferenceMember": ("conference_members", "conference", "conference_member"),
    ":name": ("persons", "id", "name"), # ("conferences", "id", "name"),
    ":Author": ("authors", "id"),
    ":Co-author": ("co_authors", "id"),
    ":Document": ("documents", "id"),
    ":PaperFullVersion": ("paper_full_versions", "id"),
    ":PaperAbstract": ("paper_abstracts", "id"),
    ":ProgramCommittee": ("program_committees", "id"),
    ":ProgramCommitteeMember": ("pc_members", "id"),
    ":ProgramCommitteeChair": ("program_committee_chairs", "id"),
    ":email": ("persons", "id", "email"),
    ":date": ("conferences", "id", "date"),
    # "rdfs:label.to": "label",
    ":paperID": ("papers", "id", "paper_id"),
    ":title": ("papers", "id", "title"),
    # "rdfs:comment.to": "comment",
    ":hasAuthor": ("papers", "id", "author"),
    ":co-writePaper": ("co_author_paper", "cid", "pid"),
    ":hasBeenAssigned": ("paper_reviewer", "rid", "pid"),
    ":hasProgramCommitteeMember": ("program_committee_members", "program_committee", "program_committee_member"),
}

problem = mappingProblem.MappingProblem(schema, ontology, mappings, use_db=False)
problem.add_query_dir(query_dir)
