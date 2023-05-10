import os

from . import DBPEDIA50_PATH, DBPEDIA50_REASONED_PATH

robot_command = f"""
sudo robot reason --reasoner hermit --input {str(DBPEDIA50_PATH / "DBpedia50.owl")} --axiom-generators "ClassAssertion" --create-new-ontology true --output {str(DBPEDIA50_REASONED_PATH / "DBpedia50_reasoned.owl")}
"""

os.system(robot_command)
