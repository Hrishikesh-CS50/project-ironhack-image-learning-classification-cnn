# train.py
import config
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.optimizers import SGD


def compile_model(model):
    optimizer = SGD(
        learning_rate=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        nesterov=True)

    model.compile(
        optimizer=optimizer,
        loss="SparseCategoricalCrossentropy",
        metrics=["accuracy"],
    )
    return model

def warmup_step_decay(epoch):
    # Warmup: gradually increase
    if epoch < 5:
        return 0.002 * (epoch + 1)   # 0.002 → 0.01
    
    # Main phase
    if epoch < 30:
        return 0.01
    
    # Decay phase 1
    if epoch < 45:
        return 0.001
    
    # Decay phase 2
    return 0.0001

lr_scheduler = LearningRateScheduler(warmup_step_decay, verbose=1)


def train_model(model, x_train, y_train, batch_size, epochs):
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,      # increased to allow LR Scheduler to take effect
        restore_best_weights=True
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[early_stopping, lr_scheduler]
    )
    return history

def evaluate_model(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return loss, accuracy
