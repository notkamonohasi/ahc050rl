from pydantic import BaseModel


class Point(BaseModel):
    y: int
    x: int

    def __add__(self, other: tuple[int, int]) -> "Point":
        return Point(y=self.y + other[0], x=self.x + other[1])
