from keras import layers, models
from keras.datasets import mnist
from skeras import plot_loss, plot_acc
import matplotlib.pyplot as plt

class AE(models.Model):
    def __init__(self, x_nodes, z_dim):
        x_shape = (x_nodes,)
        x = layers.Input(shape=x_shape) # 입력계층
        z = layers.Dense(z_dim, activation='relu')(x) # 은닉계층
        y = layers.Dense(x_nodes, activation='sigmoid')(z) # 출력계층

        super().__init__(x, y)
        self.compile(optimizer='adadelta', loss='binary_crossentropy')
        self.x = x
        self.z = z
        self.z_dim = z_dim
    
    def Encoder(self):
        return models.Model(self.x, self.z)
    
    def Decoder(self):
        z_shape = (self.z_dim,)
        z = layers.Input(shape=z_shape)
        y_layer = self.layers[-1]
        y = y_layer(z)
        return models.Model(z, y)

# 데이터 준비

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train = x_train.reshape((len(x_train), -1))
x_test = x_test.reshape((len(x_test), -1))

# 완전 연결 계층 AE 학습

x_nodes = 784
z_dim = 36

autoencoder = AE(x_nodes, z_dim)

history = autoencoder.fit(
    x_train,    # 입력과 출력은 동일한 데이터이다.
    x_train,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test))


encoder = autoencoder.Encoder()
decoder = autoencoder.Decoder()

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10  # 화면에 표시할 이미지의 수
plt.figure(figsize=(20,6)) # 그림의 전체 크기 지정

# 평가용 이미지, 부호화 그래프, 복화화 이미치 출력
for i in range(n):
    ax = plt.subplot(3, n, i+1)

    # 입력이미지
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()

    # 이미지 주변 축들은 보이지 않도록 설정
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 이미지의 압축한 형태 출력. 합성곱 AE는 2차원 이미지를 1차원의 벡터로 압축하기 때문에 부호화된 정보는 1차원 그래프로 표시
    ax = plt.subplot(3, n, i+1+n)
    plt.stem(encoded_imgs[i].reshape(-1))

    # 복호화한 이미지 출력
    ax = plt.subplot(3, n, i+1+n+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))

plt.show()