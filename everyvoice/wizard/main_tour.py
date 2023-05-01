from everyvoice.wizard import StepNames, Tour
from everyvoice.wizard.basic import MoreDatasetsStep, NameStep, OutputPathStep
from everyvoice.wizard.dataset import return_dataset_steps

WIZARD_TOUR = Tour(
    name="Basic Tour",
    steps=[
        NameStep(name=StepNames.name_step.value),
        OutputPathStep(name=StepNames.output_step.value),
    ]
    + return_dataset_steps()
    + [MoreDatasetsStep(name=StepNames.more_datasets_step.value)],
)

if __name__ == "__main__":
    WIZARD_TOUR.run()
