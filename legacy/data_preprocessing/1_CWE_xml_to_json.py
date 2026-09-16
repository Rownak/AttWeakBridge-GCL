import json
import xml.etree.ElementTree as ET
import sys
import os
# Add project root to path to import config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(project_root)
sys.path.insert(0, project_root)
import config

data_dir = config.DATA_DIR
cwe_dir = config.CWE_DIR
print(data_dir)
print(cwe_dir)

# Load the XML file
xml_file_name = config.CWE_DATASET
tree = ET.parse(cwe_dir+xml_file_name+'.xml')
root = tree.getroot()

# Get the XML root attributes and its immediate children
root_attrib = root.attrib
children_tags = [child.tag for child in root]

(root.tag, root_attrib, children_tags)

weaknesses_section = root.find('{http://cwe.mitre.org/cwe-7}Weaknesses')
weakness_entries = list(weaknesses_section)


# Parse the XML
from collections import defaultdict
namespace = {
    'cwe': 'http://cwe.mitre.org/cwe-7',
    'xhtml': 'http://www.w3.org/1999/xhtml'
}
# Extract Weakness information
weakness_info = defaultdict(list)

count3=0
for weakness in root.findall('.//cwe:Weakness', namespace):
    weakness_id = weakness.get('ID')
    weakness_name = weakness.get('Name')
    
    description = weakness.find('.//cwe:Description', namespace).text.strip() if weakness.find('.//cwe:Description', namespace) is not None else 'N/A'
    extended_description_elems = weakness.findall('.//cwe:Extended_Description/xhtml:p', namespace)
    if not extended_description_elems:
        extended_description = weakness.find('.//cwe:Extended_Description', namespace).text.strip() if weakness.find('.//cwe:Extended_Description', namespace) is not None else 'N/A'
    else:
        extended_description = ''.join(ET.tostring(e, encoding='unicode', method='text').strip() for e in extended_description_elems)
    # print("# ID #")
    # print(weakness_id)
    # print("# description #")
    # print(description)
    # print("# extended_description #")
    # print(extended_description.strip())
    modes_of_introduction = ""
    for intro in weakness.findall('.//cwe:Introduction', namespace):
        phase = intro.find('.//cwe:Phase', namespace).text.strip() if intro.find('.//cwe:Phase', namespace) is not None else 'N/A'
        #note = intro.find('.//cwe:Note', namespace).text.strip() if intro.find('.//cwe:Note', namespace) is not None else 'N/A'
        note_elems = intro.findall('.//cwe:Note/xhtml:p', namespace)
        if not note_elems:
            note = intro.find('.//cwe:Note', namespace).text.strip() if intro.find('.//cwe:Note', namespace) is not None else 'N/A'
        else:
            note = ''.join(ET.tostring(e, encoding='unicode', method='text').strip() for e in note_elems)
            
        if(phase!='N/A' and note!='N/A'):
            modes_of_introduction+=(phase+": "+note)
        elif(note!='N/A'):
            modes_of_introduction+=(". "+note)

    # print("# modes_of_introduction #")
    # print(modes_of_introduction)
    consequences = ""
    for consequence in weakness.findall('.//cwe:Consequence', namespace):
        scopes = [scope.text.strip() for scope in consequence.findall('.//cwe:Scope', namespace)]
        impacts = [impact.text.strip() for impact in consequence.findall('.//cwe:Impact', namespace)]
        note_elems = consequence.findall('.//cwe:Note/xhtml:p', namespace)
        if not note_elems:
            note = consequence.find('.//cwe:Note', namespace).text.strip() if consequence.find('.//cwe:Note', namespace) is not None else 'N/A'
        else:
            note = ''.join(ET.tostring(e, encoding='unicode', method='text').strip() for e in note_elems)
        scope_str = "Scopes: "
        for scope in scopes:
            if(scope=='N/A'):
                continue
            scope_str+=scope
        impact_str = "Impacts: "
        for impact in impacts:
            if(impact=='N/A'):
                continue
            impact_str+=impact
        if(note!='N/A'):
            consequences+=(scope_str+". "+ impact_str+". Note: "+ note)
    # print("# consequences #")
    # print(consequences)
    detection_methods = ""
    for method in weakness.findall('.//cwe:Detection_Method', namespace):
        method_name = method.find('.//cwe:Method', namespace).text.strip() if method.find('.//cwe:Method', namespace) is not None else 'N/A'
        #description = method.find('.//cwe:Description', namespace).text if method.find('.//cwe:Description', namespace) is not None else 'N/A'
        
        description_elems = method.findall('.//cwe:Description/xhtml:p', namespace)
        if not description_elems:
            description = method.find('.//cwe:Description', namespace).text.strip() if method.find('.//cwe:Description', namespace) is not None else 'N/A'
        else:
            description = ''.join(ET.tostring(e, encoding='unicode', method='text').strip() for e in description_elems)
        effectiveness = method.find('.//cwe:Effectiveness', namespace).text.strip() if method.find('.//cwe:Effectiveness', namespace) is not None else 'N/A'
        text=""
        if(method_name!='N/A'):
            text+=" Method Name: "+method_name
        if(description!='N/A'):
            text+=". Description: "+description
        detection_methods+=text
    # print("# detection_methods #")
    # print(detection_methods)
    mitigations = ""
    for mitigation in weakness.findall('.//cwe:Mitigation', namespace):
        phase = mitigation.find('.//cwe:Phase', namespace).text.strip() if mitigation.find('.//cwe:Phase', namespace) is not None else 'N/A'
        
        description = ''.join(ET.tostring(e, encoding='unicode', method='text').strip() for e in mitigation.findall('.//cwe:Description', namespace))
        mitigations+=(phase+" : "+description)
    # print("# mitigations #")
    # print(mitigations)
    
    examples = ""

    for example in weakness.findall('.//cwe:Demonstrative_Example', namespace):
        intro_text_elems = example.findall('.//cwe:Intro_Text/xhtml:p', namespace)
        if not intro_text_elems:
            intro_text = example.find('.//cwe:Intro_Text', namespace).text.strip() if example.find('.//cwe:Intro_Text', namespace) is not None else 'N/A'
        else:
            intro_text = ''.join(ET.tostring(e, encoding='unicode', method='text').strip() for e in intro_text_elems)
        body_text_elems = example.findall('.//cwe:Body_Text/xhtml:p', namespace)
        if not body_text_elems:
            body_text = example.find('.//cwe:Body_Text', namespace).text.strip() if example.find('.//cwe:Body_Text', namespace) is not None else 'N/A'
        else:
            body_text = ''.join(ET.tostring(e, encoding='unicode', method='text').strip() for e in body_text_elems)
        
        if(intro_text!='N/A' and body_text!='N/A'):
            examples+=(intro_text+". "+ body_text)
        elif(body_text!='N/A'):
            examples+=(". "+body_text)
        elif(intro_text!='N/A'):
            examples+=(". "+intro_text)
    # print("# examples #")
    # print(examples)
    # Extract Related Weaknesses
    # Extract sub-elements
    related_weaknesses = []
    for related in weakness.findall('.//cwe:Related_Weaknesses/cwe:Related_Weakness', namespace):
        related_weakness = {
            'Nature': related.get('Nature'),
            'CWE_ID': related.get('CWE_ID'),
            'View_ID': related.get('View_ID'),
            'Ordinal': related.get('Ordinal')
        }
        related_weaknesses.append(related_weakness)

    weakness_info[weakness_id] = {
        'ID':weakness_id,
        'Name': weakness_name,
        'Description': description.strip(),
        'Extended_Description': extended_description.strip(),
        'Modes_Of_Introduction': modes_of_introduction.strip(),
        'Common_Consequences': consequences.strip(),
        'Detection_Methods': detection_methods.strip(),
        'Potential_Mitigations': mitigations.strip(),
        'Demonstrative_Examples': examples.strip(),
        'Related_Weaknesses':related_weaknesses
    }

with open(cwe_dir+xml_file_name+'_info.json','w') as fp:
    json.dump(weakness_info,fp)