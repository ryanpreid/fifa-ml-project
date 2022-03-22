from ui.app import App
from ui.fifaframe import FifaFrame


if __name__ == "__main__":
    app = App()
    frame = FifaFrame(app)
    app.mainloop()
