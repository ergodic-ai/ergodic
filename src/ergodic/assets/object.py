# {'id': '2351a349-c488-46d2-9ddc-6f746dc9c164',
#   'name': 'Diabetes Dataset',
#   'userid': 'c1a48216-e60b-45a7-bae5-f0fdf5a6be4d',
#   'object_type': 'PickleContext',
#   'ref': 'c1a48216-e60b-45a7-bae5-f0fdf5a6be4d/data/2351a349-c488-46d2-9ddc-6f746dc9c164.pkl',
#   'metadata': '{"database_spec":{"description":"The diabetes database contains medical data related to diabetes patients. It includes various health metrics such as age, sex, body mass index (BMI), blood pressure (BP), and several serum measurements. This database is used for analyzing and predicting diabetes-related health outcomes.","table_names":["diabetes"],"tables":[{"description":"This table contains medical data for diabetes patients, including various health metrics and serum measurements.","column_names":["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6","target"],"columns":[{"description":"Age of the patient in years, represented as a normalized double value.","data_type":"DOUBLE","is_nullable":true,"unique_sample":[0.038075906433423,0.0417084448844424,-0.0963280162542955,-0.0236772472339071,0.0344433679824036],"name":"age"},{"description":"Sex of the patient, represented as a normalized double value.","data_type":"DOUBLE","is_nullable":true,"unique_sample":[0.0506801187398186,-0.0446416365069891],"name":"sex"},{"description":"Body Mass Index of the patient, represented as a normalized double value.","data_type":"DOUBLE","is_nullable":true,"unique_sample":[-0.0288400076873038,-0.0180618869484958,-0.022373135244019,-0.0040503299880454,-0.0126728265790918],"name":"bmi"},{"description":"Average blood pressure of the patient, represented as a normalized double value.","data_type":"DOUBLE","is_nullable":true,"unique_sample":[-0.0148628343982749,-0.1009341087945064,0.0459723423449815,-0.0527341951326168,-0.0366560810754007],"name":"bp"},{"description":"Total serum cholesterol (tc) of the patient, represented as a normalized double value.","data_type":"DOUBLE","is_nullable":true,"unique_sample":[0.0906198816792638,0.0080627101871966,0.0878679759628616,0.0053108044707943,0.0383336730676212],"name":"s1"},{"description":"Low-density lipoproteins (ldl) of the patient, represented as a normalized double value.","data_type":"DOUBLE","is_nullable":true,"unique_sample":[-0.0248000120604338,0.0062016856567301,-0.0129003705124315,-0.0285577936019082,0.014969842586837],"name":"s2"},{"description":"High-density lipoproteins (hdl) of the patient, represented as a normalized double value.","data_type":"DOUBLE","is_nullable":true,"unique_sample":[-0.0434008456520249,0.0744115640787572,-0.0286742944356771,-0.0581273968683726,0.0265502726256269],"name":"s3"},{"description":"Total cholesterol / HDL (tch) of the patient, represented as a normalized double value.","data_type":"DOUBLE","is_nullable":true,"unique_sample":[0.0343088588777267,0.0029429061332032,-0.0505637191368662,0.0560805201945136,0.0361539149215222],"name":"s4"},{"description":"Possibly the log of serum triglycerides level (ltg) of the patient, represented as a normalized double value.","data_type":"DOUBLE","is_nullable":true,"unique_sample":[0.002861309289833,-0.0307479175330982,0.0191964691668856,-0.0425708541182193,-0.015998872510179],"name":"s5"},{"description":"Blood sugar level (glu) of the patient, represented as a normalized double value.","data_type":"DOUBLE","is_nullable":true,"unique_sample":[-0.0259303389894727,-0.0466408735636449,0.135611830689071,0.0403433716478785,0.0734802269665542],"name":"s6"},{"description":"The target variable, representing the diabetes progression measure, represented as a double value.","data_type":"DOUBLE","is_nullable":true,"unique_sample":[135.0,144.0,168.0,68.0,137.0],"name":"target"}]}],"links":[]},"table_name":"diabetes","user_context":{"name":"diabetes","context":"age age in years\\nsex\\nbmi body mass index\\nbp average blood pressure\\ns1 tc, total serum cholesterol\\ns2 ldl, low-density lipoproteins\\ns3 hdl, high-density lipoproteins\\ns4 tch, total cholesterol / HDL\\ns5 ltg, possibly log of serum triglycerides level\\ns6 glu, blood sugar level"},"user_metadata":null}',
#   'user_context': {'name': 'diabetes',
#    'context': 'age age in years\nsex\nbmi body mass index\nbp average blood pressure\ns1 tc, total serum cholesterol\ns2 ldl, low-density lipoproteins\ns3 hdl, high-density lipoproteins\ns4 tch, total cholesterol / HDL\ns5 ltg, possibly log of serum triglycerides level\ns6 glu, blood sugar level'},
#   'inputs': [],
#   'logger': '{"history":[]}',
#   'version': '0.0.1',
#   'createdat': '2024-09-09T17:39:50.226000Z',
#   'updatedat': '2024-09-09T17:39:50.226000Z',
#   'description': 'The diabetes database contains medical data related to diabetes patients. It includes various health metrics such as age, sex, body mass index (BMI), blood pressure (BP), and several serum measurements. This database is used for analyzing and predicting diabetes-related health outcomes.',
#   'profiles': None,
#   'kg_edges': None,
#   'kg_nodes': None,
#   'orbitasset': None}

from typing import List
from pydantic import BaseModel
from datetime import datetime
from abc import ABC, abstractmethod


class Asset(BaseModel, ABC):
    """Abstract class for an asset."""

    id: str
    name: str
    userid: str
    object_type: str
    description: str
    createdat: datetime
    updatedat: datetime
    inputs: List[str]

    def __init__(self, **kwargs):
        id = kwargs.get("id")
        name = kwargs.get("name")
        userid = kwargs.get("userid")
        object_type = kwargs.get("object_type")
        description = kwargs.get("description")
        createdat = kwargs.get("createdat")
        updatedat = kwargs.get("updatedat")
        inputs = kwargs.get("inputs")

        super().__init__(
            id=id,
            name=name,
            userid=userid,
            object_type=object_type,
            description=description,
            createdat=createdat,
            updatedat=updatedat,
            inputs=inputs,
        )

    def __str__(self):
        return f"""Asset(id={self.id}, name={self.name}, userid={self.userid}, object_type={self.object_type}, 
        description={self.description}, createdat={self.createdat}, updatedat={self.updatedat}, inputs={self.inputs})"""

    @abstractmethod
    def query(self, query: str, **kwargs):
        pass

    @abstractmethod
    def get_context(self, **kwargs):
        pass
