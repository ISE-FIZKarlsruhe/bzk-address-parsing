from dataclasses import dataclass
from modules.pipeline.linked_data import LinkedAddress
from modules.pipeline.linking_steps import LinkingStepResult, Success

@dataclass(frozen=True)
class LinkingStepMetadata:
    step_name : str
    result : LinkingStepResult
    start_epoch : float
    elapsed_time_seconds : float
    
    @property
    def was_successful(self) -> bool:
        return isinstance(self.result, Success)

@dataclass(frozen=True)
class AddressLinkingMetadata:
    address : LinkedAddress
    applied_steps : list[LinkingStepMetadata]

    @property
    def total_elapsed_time_seconds(self) -> float:
        return sum(step.elapsed_time_seconds for step in self.applied_steps)

    def minimize(self) -> "AddressLinkingMetadata":
        """
        Address object keeps all matches, 
        it may be desirable to drop them after disambiguation
        """
        minimized_entities = []
        for entity in self.address.entities:
            minimized_entities.append(entity.minimize())
        minimized_address = self.address.copy(update={"entities": minimized_entities})
        return AddressLinkingMetadata(
            address=minimized_address,
            applied_steps=self.applied_steps
        )