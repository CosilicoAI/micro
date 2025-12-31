"""Variable definitions and ontology.

Variables are the atomic units of data, linked to legal definitions
and source datasets.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from microplex.core.entities import EntityType
from microplex.core.periods import PeriodType


class DataType(Enum):
    """Data types for variables."""

    MONEY = "money"
    RATE = "rate"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    DATE = "date"
    STRING = "string"

    @property
    def precision(self) -> int | None:
        """Decimal precision for numeric types."""
        return {
            DataType.MONEY: 2,
            DataType.RATE: 6,
            DataType.FLOAT: 6,
        }.get(self)

    @property
    def min_value(self) -> float | None:
        """Minimum value for bounded types."""
        return {
            DataType.RATE: 0.0,
        }.get(self)

    @property
    def max_value(self) -> float | None:
        """Maximum value for bounded types."""
        return {
            DataType.RATE: 1.0,
        }.get(self)

    @property
    def numpy_dtype(self) -> str:
        """Numpy dtype string."""
        return {
            DataType.MONEY: "float64",
            DataType.RATE: "float64",
            DataType.FLOAT: "float64",
            DataType.BOOLEAN: "bool",
            DataType.INTEGER: "int64",
            DataType.CATEGORICAL: "object",
            DataType.DATE: "datetime64[ns]",
            DataType.STRING: "object",
        }[self]


class VariableRole(Enum):
    """Role of a variable in the system."""

    INPUT = "input"  # Observable from surveys/admin data
    OUTPUT = "output"  # Calculated by policy engine
    INTERMEDIATE = "intermediate"  # Calculated but not final output
    PARAMETER = "parameter"  # Policy parameter (not person-level)

    @property
    def is_observable(self) -> bool:
        """Whether this role is directly observable in data."""
        return self in (VariableRole.INPUT, VariableRole.OUTPUT)


class LegalReference(BaseModel):
    """Reference to legal authority for a variable definition."""

    source: str = Field(..., description="USC, CFR, SSA, etc.")
    title: int | None = None
    part: int | None = None
    section: int | str | None = None
    subsection: str | None = None
    paragraph: str | None = None

    model_config = {"frozen": True}

    @classmethod
    def usc(cls, title: int, section: int, subsection: str | None = None) -> LegalReference:
        """Create USC reference."""
        return cls(source="USC", title=title, section=section, subsection=subsection)

    @classmethod
    def cfr(cls, title: int, part: int, section: int, subsection: str | None = None) -> LegalReference:
        """Create CFR reference."""
        return cls(source="CFR", title=title, part=part, section=section, subsection=subsection)

    @property
    def url(self) -> str:
        """URL to legal text."""
        if self.source == "USC":
            return f"https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title{self.title}-section{self.section}"
        elif self.source == "CFR":
            return f"https://www.ecfr.gov/current/title-{self.title}/part-{self.part}/section-{self.part}.{self.section}"
        return ""

    def __str__(self) -> str:
        if self.source == "USC":
            s = f"{self.title} USC {self.section}"
            if self.subsection:
                s = f"{self.title} USC {self.section}{self.subsection}"
            return s
        elif self.source == "CFR":
            return f"{self.title} CFR {self.part}.{self.section}"
        return f"{self.source} {self.section}"


class Variable(BaseModel):
    """A variable in the microdata schema.

    Variables are the atomic units of data. Each has:
    - A name (identifier)
    - An entity type (person, tax_unit, household)
    - A data type (money, rate, boolean, etc.)
    - Optional legal references linking to statutes
    - Optional source mappings to survey/admin data
    """

    name: str = Field(..., description="Variable identifier")
    entity: EntityType = Field(..., description="Entity level")
    dtype: DataType = Field(..., description="Data type")
    period: PeriodType = Field(default=PeriodType.YEAR, description="Default period")
    role: VariableRole = Field(default=VariableRole.INPUT, description="Variable role")

    # Metadata
    label: str | None = Field(default=None, description="Human-readable label")
    description: str | None = Field(default=None, description="Long description")
    unit: str | None = Field(default=None, description="Unit (USD, %, count)")

    # Legal references
    legal_references: list[LegalReference] = Field(default_factory=list)

    # Source mappings
    sources: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of source name to variable name in that source",
    )

    # Validation
    categories: list[str] | None = Field(
        default=None,
        description="Allowed values for categorical variables",
    )
    min_value: float | None = None
    max_value: float | None = None

    # Dependencies
    dependencies: list[str] = Field(
        default_factory=list,
        description="Variables this depends on for calculation",
    )

    # Uncertainty
    uncertainty_cv: float | None = Field(
        default=None,
        description="Coefficient of variation for uncertainty",
    )

    model_config = {"frozen": True}

    def validate_value(self, value: Any) -> bool:
        """Check if a value is valid for this variable."""
        # Type check
        if self.dtype == DataType.BOOLEAN:
            if not isinstance(value, bool):
                return False
        elif self.dtype == DataType.INTEGER:
            if not isinstance(value, int) or isinstance(value, bool):
                return False
        elif self.dtype in (DataType.MONEY, DataType.RATE, DataType.FLOAT):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False
        elif self.dtype == DataType.CATEGORICAL:
            if self.categories and value not in self.categories:
                return False
        elif self.dtype == DataType.STRING:
            if not isinstance(value, str):
                return False

        # Range check
        if self.dtype == DataType.RATE:
            if value < 0.0 or value > 1.0:
                return False
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False

        return True


class VariableRegistry:
    """Registry of all variables in the system."""

    def __init__(self) -> None:
        self._variables: dict[str, Variable] = {}

    def register(self, variable: Variable) -> None:
        """Register a variable."""
        self._variables[variable.name] = variable

    def get(self, name: str) -> Variable | None:
        """Get variable by name."""
        return self._variables.get(name)

    def __getitem__(self, name: str) -> Variable:
        """Get variable by name, raising if not found."""
        var = self.get(name)
        if var is None:
            raise KeyError(f"Variable not found: {name}")
        return var

    def __contains__(self, name: str) -> bool:
        return name in self._variables

    def __iter__(self):
        return iter(self._variables.values())

    def __len__(self) -> int:
        return len(self._variables)

    def by_entity(self, entity: EntityType) -> list[Variable]:
        """Get all variables for an entity type."""
        return [v for v in self._variables.values() if v.entity == entity]

    def by_source(self, source: str) -> list[Variable]:
        """Get all variables that have a mapping for a source."""
        return [v for v in self._variables.values() if source in v.sources]

    def by_role(self, role: VariableRole) -> list[Variable]:
        """Get all variables with a given role."""
        return [v for v in self._variables.values() if v.role == role]

    @classmethod
    def from_yaml(cls, path: str | Path) -> VariableRegistry:
        """Load variables from YAML file."""
        registry = cls()
        path = Path(path)

        with open(path) as f:
            data = yaml.safe_load(f)

        for name, spec in data.items():
            # Parse entity type
            entity_str = spec.get("entity", "person").lower()
            entity = EntityType(entity_str)

            # Parse data type
            dtype_str = spec.get("dtype", "float").lower()
            dtype = DataType(dtype_str)

            # Parse period type
            period_str = spec.get("period", "year").lower()
            period = PeriodType(period_str)

            # Parse legal references
            legal_refs = []
            for ref_spec in spec.get("legal_references", []):
                ref_type = ref_spec.get("type", "usc").upper()
                if ref_type == "USC":
                    legal_refs.append(
                        LegalReference.usc(
                            ref_spec["title"],
                            ref_spec["section"],
                            ref_spec.get("subsection"),
                        )
                    )
                elif ref_type == "CFR":
                    legal_refs.append(
                        LegalReference.cfr(
                            ref_spec["title"],
                            ref_spec["part"],
                            ref_spec["section"],
                            ref_spec.get("subsection"),
                        )
                    )

            variable = Variable(
                name=name,
                entity=entity,
                dtype=dtype,
                period=period,
                label=spec.get("label"),
                description=spec.get("description"),
                sources=spec.get("sources", {}),
                legal_references=legal_refs,
                categories=spec.get("categories"),
                min_value=spec.get("min_value"),
                max_value=spec.get("max_value"),
                dependencies=spec.get("dependencies", []),
                uncertainty_cv=spec.get("uncertainty_cv"),
            )
            registry.register(variable)

        return registry
