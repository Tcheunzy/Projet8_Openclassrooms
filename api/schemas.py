"""Schémas Pydantic pour la validation des données de l'API."""
from typing import Literal, Optional, Any
from pydantic import BaseModel, ConfigDict, Field
import pandas as pd

class ClientPredictionInput (BaseModel):
    """ Données minimales nécéssaires  au fonctionnement de l'API (les données les plus impactantes sur 
    la prise de décision du modèle sur les 432 features utilisées pour la prédiction)"""

    SK_ID_CURR: int = Field(gt=0)
    AMT_INCOME_TOTAL: float = Field(gt=0)
    AMT_CREDIT: float = Field(gt=0)
    AMT_ANNUITY: float = Field(gt=0)
    AMT_GOODS_PRICE: float = Field(gt=0)
    DAYS_BIRTH: int = Field(ge=-25550, le=-6570)
    DAYS_EMPLOYED: int = Field (le=0)
    CNT_FAM_MEMBERS: float = Field(ge=1)
    CNT_CHILDREN: int = Field(ge=0)
    CODE_GENDER: Literal["F", "M", "XNA"]
    NAME_CONTRACT_TYPE : Literal["Cash loans", "Revolving loans"]
    FLAG_OWN_CAR : Literal["Y", "N"]
    FLAG_OWN_REALTY : Literal["Y", "N"]
    NAME_EDUCATION_TYPE : Literal['Lower secondary', 'Secondary / secondary special', 'Incomplete higher','Higher education', 'Academic degree']
    EXT_SOURCE_1 : Optional[float] = Field(None,ge=0, le=1)
    EXT_SOURCE_2 : Optional[float] = Field(None,ge=0, le=1)
    EXT_SOURCE_3 : Optional[float] = Field(None,ge=0, le=1)
    NAME_INCOME_TYPE : Optional[str] = Field(None)
    NAME_FAMILY_STATUS : Optional[str] = Field(None)
    NAME_HOUSING_TYPE : Optional[str] = Field(None)
    OCCUPATION_TYPE : Optional[str] = Field(None)
    ORGANIZATION_TYPE : Optional[str] = Field(None)
    REGION_POPULATION_RELATIVE : Optional[float] = Field(None)
    extra_fields: dict[str, Any] = Field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        data = self.model_dump(exclude={"extra_fields"})
        data.update(self.extra_fields)
        return pd.DataFrame([data])

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "SK_ID_CURR":827,
                "AMT_INCOME_TOTAL":150000,
                "AMT_CREDIT": 500000,
                "AMT_ANNUITY": 1000,
                "AMT_GOODS_PRICE": 5000,
                "DAYS_BIRTH" : -16000,
                "DAYS_EMPLOYED" : -400,
                "CNT_FAM_MEMBERS" : 1,
                "CNT_CHILDREN" : 0,
                "CODE_GENDER": "F",
                "NAME_CONTRACT_TYPE": "Cash loans",
                "FLAG_OWN_CAR": "Y",
                "FLAG_OWN_REALTY": "Y",
                "NAME_EDUCATION_TYPE": "Lower secondary",
                "EXT_SOURCE_1": 0.59,
                "EXT_SOURCE_2": 0.78,
                "EXT_SOURCE_3": 0.32,



            }
        }
    )


class PredictionResponse (BaseModel):
    """ Schema pydantic des outputs du modèle"""

    sk_id_curr : int 
    probability: float = Field(ge=0, le=1)
    threshold : float
    decision : Literal["accordé", "refusé"]
    mlflow_model_version: str

    model_config = ConfigDict(
        json_schema_extra={
            "example" :{
                "sk_id_curr" :12232,
                "probability" : 0.88,
                "threshold" : 0.24,
                "decision" : "refusé",
                "mlflow_model_version" : "4"
            }
        }
    )






