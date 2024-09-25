from everyvoice.wizard import StepNames, Tour
from everyvoice.wizard.basic import (
    ContactEmailStep,
    ContactNameStep,
    MoreDatasetsStep,
    NameStep,
    OutputPathStep,
)
from everyvoice.wizard.dataset import get_dataset_steps


def get_main_wizard_tour(trace: bool = False, debug_state: bool = False) -> Tour:
    """Get the main wizard tour"""
    return Tour(
        name="Basic Tour",
        steps=[
            NameStep(name=StepNames.name_step),
            ContactNameStep(name=StepNames.contact_name_step),
            ContactEmailStep(name=StepNames.contact_email_step),
            OutputPathStep(name=StepNames.output_step),
        ]
        + get_dataset_steps()
        + [MoreDatasetsStep(name=StepNames.more_datasets_step)],
        trace=trace,
        debug_state=debug_state,
    )


if __name__ == "__main__":
    get_main_wizard_tour().run()
