# Computer Vision Задание 2
На основе ноутбука из предыдущего задания, свернуть нейросеть, используя минимум два уровня свертывания.

Слои в нашей модели.
```python
model = keras.Sequential([
                          keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
                          keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
                          keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
                          keras.layers.Conv2D(128, (3, 3), padding='same', activation="relu"),
                          keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
                          keras.layers.Conv2D(128, (3, 3), padding='same', activation="relu"),
                          keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
                          keras.layers.Flatten(),
                          keras.layers.Dense(128, activation='relu'),
                          keras.layers.Dense(10, activation="softmax")
])
```
Параметры компиляции модели:
```python
model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.002), loss='MeanSquaredLogarithmicError', metrics=['accuracy'])
```
Результаты обучения модели:
```python
test_loss, test_acc = model.evaluate(x_test, y_test_cat)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
```
<code>313/313 [==============================] - 24s 75ms/step - loss: 0.0168 - accuracy: 0.7702
Test loss: 0.016804629936814308
Test accuracy: 0.7702000141143799
</code>

![image info](https://github.com/misshp11/CV3/blob/main/img/изображение_2023-04-19_210943936.png)  
![image info](https://github.com/misshp11/CV3/blob/main/img/изображение_2023-04-19_210856080.png)  
![image info](https://github.com/misshp11/CV3/blob/main/img/изображение_2023-04-19_210827761.png)  
![image info](https://github.com/misshp11/CV3/blob/main/img/изображение_2023-04-19_210915776.png)  
