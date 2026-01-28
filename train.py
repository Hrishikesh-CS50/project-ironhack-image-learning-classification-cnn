# train.py
def compile_model(model):
    model.compile(
        optimizer="SGD",
        loss="SparseCategoricalCrossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(model, x_train, y_train, batch_size, epochs):
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
    )
    model.save("models/model.keras")
    return history


def evaluate_model(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return loss, accuracy