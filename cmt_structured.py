import rdflib
import os

import mappingProblem
import query
import util

schema = "cmt_structured"
ontology = "/users/zsoori/mapping/RODI/data/cmt_structured/ontology.ttl"
query_dir = "/users/zsoori/mapping/RODI/data/cmt_structured/queries"

mappings = {
    ":Conference": "conferences",
    ":siteURL.to": "site_url",
    ":Review": "reviews",
    ":Reviewer": "reviewers",
    ":writeReview.to": "written",
    ":Person": '"Person"',
    ":hasConferenceMember": "conference_members",
    ":hasConferenceMember.from": "conference",
    ":hasConferenceMember.to": "conference_member",
    ":name.to": "name",
    ":Author": "authors",
    ":Co-author": "co_authors",
    ":Document": "documents",
    ":PaperFullVersion": "paper_full_versions",
    ":PaperAbstract": "paper_abstracts",
    ":ProgramCommittee": "program_committees",
    ":ProgramCommitteeMember": "pc_members",
    ":ProgramCommitteeChair": "program_committee_chairs",
    ":name.to": "name",
    ":email.to": "email",
    ":date.to": "date",
    "rdfs:label.to": "label",
    ":paperID": "papers",
    ":paperID.from": "id",
    ":paperID.to": "paper_id",
    ":title": "papers",
    ":title.from": "id",
    ":title.to": "title",
    "rdfs:comment.to": "comment",
    ":hasAuthor": "papers",
    ":hasAuthor.from": "id",
    ":hasAuthor.to": "author",
    ":co-writePaper": "co_author_paper",
    ":co-writePaper.from": "cid",
    ":co-writePaper.to": "pid",
    ":hasBeenAssigned": "paper_reviewer",
    ":hasBeenAssigned.from": "rid",
    ":hasBeenAssigned.to": "pid",
    ":hasProgramCommitteeMember": "program_committee_members",
    ":hasProgramCommitteeMember.from": "program_committee",
    ":hasProgramCommitteeMember.to": "program_committee_member",
}

problem = mappingProblem.MappingProblem(schema, ontology, mappings)

for filename in os.listdir(query_dir):
    if filename.endswith(".qpair"):
        fullname = os.path.join(query_dir, filename)
        print(fullname)
        q = query.Query(fullname)
        problem.queries.append(q)
problem.update_namespaces()
problem.queries.sort(key=lambda x: x.filename)

