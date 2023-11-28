# [[FEATURE] Support Pydantic Models as ParamTypes](https://github.com/tiangolo/typer/issues/111)
import click
import typer
from pydantic import BaseModel

app = typer.Typer()


class User(BaseModel):
    id: int
    name: str = "Jane Doe"


class UserParamType(click.ParamType):
    def convert(self, value, param, ctx):
        return User.parse_raw(value)


USER = UserParamType()


# def main(num: int, user: USER):
@app.command()
def main(num: int, user: USER):
    print(num, type(num))
    print(user, type(user))


if __name__ == "__main__":
    # typer.run(main)
    app()
