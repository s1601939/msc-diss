# Import libraries
from xml.etree.ElementTree import parse as xml_parse
from pprint import pprint

# Read in the xml file with the parsed story
e = xml_parse("story.txt.xml").getroot()

# path_tags is a list of tuples: [(tag, attribute, attribute_id), ...]
def get_xml_struct(root, path_tags):
    current = root
    for tag, attribute, attribute_value in path_tags:
        for c in current:
            if c.tag == tag:
                if attribute is not None and attribute_value is not None:
                    if c.attrib.get(attribute) == attribute_value:
                        current = c
                        break
                else:
                    current = c
                    break
    return current
​
def get_attrib_value(root, dep_type, dependent=True):
    identifiers = {}
    for c in root:
        if c.tag == "dep":
            if c.attrib.get("type", "") == dep_type:
                for d in c:
                    if dependent and d.tag == "dependent":
                        return d.attrib.get("idx", "")
                    identifiers[d.tag] = d.attrib.get("idx", "")
    return identifiers
​
def get_token_lemma(tokens, token_id):
    for t in tokens:
        if t.tag == "token" and t.attrib.get("id", "") == token_id:
            for i in t:
                if i.tag == "lemma":
                    return i.text
    return "none"

# get sentences ids
sentence_ids = []
t = get_xml_struct(e, [("document", None, None), ("sentences", None, None)])
for c in t:
    id = c.attrib.get("id")
    if id is not None:
        sentence_ids.append(id)
​
print "Sentence ids:"
print sentence_ids

sentence_dependencies = {}
for sid in sentence_ids:
    sentence_structure = {"root":"none", "nsubj":"none", "dobj":"none", "nmod":"none"}
    # get basic-dependencies
    basicd = get_xml_struct(e, [("document", None, None), ("sentences", None, None), \
                                ("sentence", "id", sid), ("dependencies", "type", "basic-dependencies")
                               ]
                           )
    # get lemmas
    tokens = get_xml_struct(e, [("document", None, None), ("sentences", None, None), \
                                ("sentence", "id", sid), ("tokens", None, None)
                               ]
                           )
​
    for i in ["nsubj", "nsubjpass"]:
        it = get_token_lemma(tokens, get_attrib_value(basicd, i)).lower()
        if it != "none":
            sentence_structure["nsubj"] = it
    for i in ["root", "dobj", "nmod"]:
        it = get_token_lemma(tokens, get_attrib_value(basicd, i)).lower()
        if it != "none":
            sentence_structure[i] = it
    sentence_dependencies[sid] = sentence_structure
​
print sentence_dependencies

predicates_string = ""
for i in sorted(sentence_dependencies.keys()):
    si = sentence_dependencies[i]
    si["time"] = i
    
    x = "s(1) :: %(root)s(%(nsubj)s, %(dobj)s, %(nmod)s) at %(time)s." % si
    
    x = x.lower()
    predicates_string += x + "\n"

print predicates_string,

# Save the rules to the `extracted_knowledge.pl` file
with open("extracted_knowledge.pl", "w") as rules_file:
    rules_file.write(predicates_string)