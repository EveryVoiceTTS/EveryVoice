from typing import Optional

from clidantic import Parser
from pydantic import BaseModel


class Arguments(BaseModel):
    field_a: str
    field_b: int
    field_c: Optional[bool] = False


cli = Parser()


@cli.command()
def main(args: Arguments):
    print(args)


if __name__ == "__main__":
    cli()
