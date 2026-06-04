from dataclasses import dataclass, field
from typing import List, Dict, Any
import yaml
import os

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

def load_circuit_yaml(file_path: str) -> CircuitConfig:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    elements = [Element(**el) for el in data.get('elements', [])]
    
    config = CircuitConfig(
        name=data['name'],
        reference=data['reference'],
        nodes=data['nodes'],
        accessible=data['accessible'],
        elements=elements
    )
    
    validate_config(config)
    return config

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
        if el.type not in ['R']:
            raise ValueError(f"Element {el.name} has unsupported type '{el.type}'. Only 'R' is supported in MVP.")
            
        if el.n1 not in config.nodes or el.n2 not in config.nodes:
            raise ValueError(f"Element {el.name} refers to nodes not in nodes list.")
        if el.n1 == el.n2:
            raise ValueError(f"Element {el.name} has identical nodes {el.n1} and {el.n2}.")
        if el.value <= 0:
            raise ValueError(f"Element {el.name} must have positive value (conductance).")
