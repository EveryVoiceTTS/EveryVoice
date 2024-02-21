from everyvoice.wizard import StepNames, Tour
from everyvoice.wizard.basic import (
    ContactEmailStep,
    ContactNameStep,
    MoreDatasetsStep,
    NameStep,
    OutputPathStep,
)
from everyvoice.wizard.dataset import return_dataset_steps

WIZARD_TOUR = Tour(
    name="Basic Tour",
    steps=[
        NameStep(name=StepNames.name_step.value),
        ContactNameStep(name=StepNames.contact_name_step.value),
        ContactEmailStep(name=StepNames.contact_email_step.value),
        OutputPathStep(name=StepNames.output_step.value),
    ]
    + return_dataset_steps()
    + [MoreDatasetsStep(name=StepNames.more_datasets_step.value)],
)

if __name__ == "__main__":
    WIZARD_TOUR.run()
