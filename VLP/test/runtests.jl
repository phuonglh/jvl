using VLP

server = @async VLP.run()

primes = Client.listPrimes(1, 30)
@info primes

primes = Client.listPrimes(1, 100)
@info primes

tokens = Client.tokenize("Giáo sư Nguyễn Gia Bình, Tổ trưởng Hội chẩn bệnh nhân Covid-19 nặng, nhận định đợt dịch này bệnh nhân nặng không nhiều, nhưng khi khó thở thì trở nặng rất nhanh do tổn thương phổi.")
@info tokens
