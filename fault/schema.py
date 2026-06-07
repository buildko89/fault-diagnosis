from dataclasses import dataclass, field
from typing import List
import yaml

# Supported element types and the meaning of `Element.value`:
#   R -> conductance G [S]  (admittance: G,           frequency-independent)
#   C -> capacitance C [F]  (admittance: j*omega*C)
#   L -> inductance  L [H]  (admittance: 1/(j*omega*L))
SUPPORTED_TYPES = ('R', 'C', 'L')
REACTIVE_TYPES = ('C', 'L')

@dataclass
class Element:
    name: str
    type: str
    n1: int
    n2: int
    value: float

@dataclass
class CircuitConfig:
    name: str
    reference: int
    nodes: List[int]
    accessible: List[int]
    elements: List[Element] = field(default_factory=list)
    # Measurement frequencies in Hz. Empty -> DC (only valid when all elements
    # are resistors). Reactive elements (C/L) require at least one positive
    # frequency. Internally converted to angular frequency omega = 2*pi*f.
    frequencies: List[float] = field(default_factory=list)

def circuit_config_from_dict(data: dict) -> CircuitConfig:
    """Build and validate a CircuitConfig from an already-parsed mapping.

    Shared by ``load_circuit_yaml`` (file path) and ``load_circuit_yaml_text``
    (in-memory YAML, e.g. a GUI upload) so the parsing/validation logic lives in
    one place.
    """
    elements = [Element(**el) for el in data.get('elements', [])]

    config = CircuitConfig(
        name=data['name'],
        reference=data['reference'],
        nodes=data['nodes'],
        accessible=data['accessible'],
        elements=elements,
        frequencies=data.get('frequencies', [])
    )

    validate_config(config)
    return config


def load_circuit_yaml(file_path: str) -> CircuitConfig:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return circuit_config_from_dict(data)


def load_circuit_yaml_text(text: str) -> CircuitConfig:
    """Load a CircuitConfig from a YAML string (e.g. a GUI file upload)."""
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("YAML content must be a mapping of circuit fields.")
    return circuit_config_from_dict(data)

def validate_config(config: CircuitConfig):
    if config.reference not in config.nodes:
        raise ValueError(f"Reference node {config.reference} not in nodes list.")
    
    # Check that there is at least one accessible node other than the reference
    accessible_free = [n for n in config.accessible if n != config.reference]
    if len(accessible_free) == 0:
        raise ValueError("At least one accessible node other than the reference node must be specified.")
    
    for node in config.accessible:
        if node not in config.nodes:
            raise ValueError(f"Accessible node {node} not in nodes list.")
            
    # Element name uniqueness
    names = [el.name for el in config.elements]
    if len(names) != len(set(names)):
        raise ValueError("Element names must be unique.")

    for el in config.elements:
        # Supported types
        if el.type not in SUPPORTED_TYPES:
            raise ValueError(
                f"Element {el.name} has unsupported type '{el.type}'. "
                f"Supported types: {', '.join(SUPPORTED_TYPES)}."
            )

        if el.n1 not in config.nodes or el.n2 not in config.nodes:
            raise ValueError(f"Element {el.name} refers to nodes not in nodes list.")
        if el.n1 == el.n2:
            raise ValueError(f"Element {el.name} has identical nodes {el.n1} and {el.n2}.")
        if el.value <= 0:
            raise ValueError(
                f"Element {el.name} must have a positive value "
                f"(conductance for R, capacitance for C, inductance for L)."
            )

    # Frequency validation.
    for f in config.frequencies:
        if f <= 0:
            raise ValueError(f"Frequencies must be positive (got {f}).")

    # Reactive elements (C/L) are frequency-dependent and cannot be evaluated at
    # DC; require at least one measurement frequency when any are present.
    has_reactive = any(el.type in REACTIVE_TYPES for el in config.elements)
    if has_reactive and not config.frequencies:
        reactive_names = [el.name for el in config.elements if el.type in REACTIVE_TYPES]
        raise ValueError(
            "Reactive elements (C/L) require at least one positive frequency in "
            f"'frequencies'; offending elements: {reactive_names}."
        )
