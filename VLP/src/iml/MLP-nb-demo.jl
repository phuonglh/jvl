### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 95332d32-fe64-11eb-29b6-59627a0e39d3
using MLDatasets

# ╔═╡ cf1fe910-55fe-43ae-9368-11a9d6ab6efa
using ImageView

# ╔═╡ a71a8ece-0a63-4129-bdf7-ef0add2ef81c
using Images

# ╔═╡ a5018eca-52c8-408c-8ff1-caf5cef45945
using Flux

# ╔═╡ 13992041-8205-4ef0-8577-d69fb23dacd1
X_train, y_train = MNIST.traindata()

# ╔═╡ aba53eb3-3fa3-4c87-b5a1-085ce946e13d
X_train[:,:,1]

# ╔═╡ ab24c367-0673-450b-aec1-0f68df6d906d
y_train[1]

# ╔═╡ 40505c32-3d19-4c2c-95f5-a3df1432e643
X_test, y_test = MNIST.testdata()

# ╔═╡ 9d684cd0-7b6d-4518-9edc-9a7f42d9e408
y_test

# ╔═╡ 7b9fab16-4aef-4f81-8edd-1b08a09a904d
size(X_test)

# ╔═╡ b7e9b315-115b-414e-97a7-c843df36ed94
size(X_train)

# ╔═╡ 8393f758-e0c8-4af4-88b3-c360afe7d363
img = X_train[:,:,1]

# ╔═╡ 43c52fbc-2638-48f3-a96a-691881f8576c
imshow(img)

# ╔═╡ cfb8edbd-8739-4a5c-acde-5b04432c653e
begin
	struct BWImage
		data::Array{UInt8, 2}
		zoom::Int
	end
	function BWImage(data::Array{T, 2}; zoom::Int=1) where T <: Real
		BWImage(floor.(UInt8, clamp.(((data .- minimum(data)) / (maximum(data) .- minimum(data))) * 255, 0, 255)), zoom)
	end
	
	import Base: show
	
	function show(io::IO, ::MIME"image/bmp", i::BWImage)

		orig_height, orig_width = size(i.data)
		height, width = (orig_height, orig_width) .* i.zoom
		datawidth = Integer(ceil(width / 4)) * 4

		bmp_header_size = 14
		dib_header_size = 40
		palette_size = 256 * 4
		data_size = datawidth * height * 1

		# BMP header
		write(io, 0x42, 0x4d)
		write(io, UInt32(bmp_header_size + dib_header_size + palette_size + data_size))
		write(io, 0x00, 0x00)
		write(io, 0x00, 0x00)
		write(io, UInt32(bmp_header_size + dib_header_size + palette_size))

		# DIB header
		write(io, UInt32(dib_header_size))
		write(io, Int32(width))
		write(io, Int32(-height))
		write(io, UInt16(1))
		write(io, UInt16(8))
		write(io, UInt32(0))
		write(io, UInt32(0))
		write(io, 0x12, 0x0b, 0x00, 0x00)
		write(io, 0x12, 0x0b, 0x00, 0x00)
		write(io, UInt32(0))
		write(io, UInt32(0))

		# color palette
		write(io, [[x, x, x, 0x00] for x in UInt8.(0:255)]...)

		# data
		padding = fill(0x00, datawidth - width)
		for y in 1:orig_height
			for z in 1:i.zoom
				line = vcat(fill.(i.data[y,:], (i.zoom,))...)
				write(io, line, padding)
			end
		end
	end
end

# ╔═╡ 7eeac53f-8b55-4e39-ab27-fdc3ee13d1a0
BWImage(img, zoom=10)

# ╔═╡ b0d286c6-9203-44fd-865e-f957df8d5b2c
img2 = X_train[:,:,6]

# ╔═╡ 0466dc26-b7ad-4e33-b892-cda11b3c2a3f
BWImage(img2, zoom=1)

# ╔═╡ 08da34b6-cdda-478e-8070-5f1a103b8866
y_train[6]

# ╔═╡ 2c86c151-729f-471f-bba8-3a07f9e98cb5
imshow(X_train[:,:,6], flipy=true)

# ╔═╡ 4837cd91-cdec-44ba-b59b-301946aa0874
unique(y_test)

# ╔═╡ 466d1699-c75b-443f-b823-596c6c8362b9
layer = Dense(10, 5, σ)

# ╔═╡ 886243c7-e507-47f1-882d-7da022cab579
x = rand(10)

# ╔═╡ c5dd42ef-613e-4308-a505-125d58e6233f
a = layer(x)

# ╔═╡ dbcace88-ee69-4fab-a6c7-a6ae9c1c1a45
layer1 = Dense(28*28, 50, σ)

# ╔═╡ 8678bccb-484f-4876-bfa7-418bee26f0b6
a1 = layer1(vec(X_train[:,:,1]))

# ╔═╡ 840131bd-0a88-41da-a8d2-0dcc38bfd8e5
layer2 = Dense(50, 32)

# ╔═╡ 14b395e7-f6c5-4dff-bc5e-d7c6b20688e4
a2 = layer2(a1)

# ╔═╡ 2280d962-58fc-44e1-9e77-1a4428611204
layer3 = Dense(32, 10)

# ╔═╡ 86d679f0-aa36-456c-af99-c8cc150200b4
a3 = layer3(a2)

# ╔═╡ 8a9202d2-6650-4ed2-9602-deecb91196d7
model = Chain(layer1, layer2, layer3)

# ╔═╡ 02c7ff3e-babe-41b2-acba-eae2393141ab
model(vec(X_train[:,:,1]))

# ╔═╡ 8ba29a81-385e-43e7-9f7e-0b709159e786


# ╔═╡ 2459937e-a35e-4f81-ab16-f191ee0963e0


# ╔═╡ eca020dc-712e-4dbd-b14d-a156ed87c9d5


# ╔═╡ f93b239b-552e-4371-87de-b63245b4a001


# ╔═╡ 1fae2aab-0f7b-4071-928a-317c6fa2d425


# ╔═╡ c7bd78bf-70a4-4bb7-a664-e646c6dbf8c8


# ╔═╡ c9e3fb05-bc3e-42c9-9331-88bfae4b9d59
Flux.onehot(s, vocab)

# ╔═╡ 651c2d51-953b-4de8-802b-2c8affb79bf0


# ╔═╡ Cell order:
# ╠═95332d32-fe64-11eb-29b6-59627a0e39d3
# ╠═13992041-8205-4ef0-8577-d69fb23dacd1
# ╠═aba53eb3-3fa3-4c87-b5a1-085ce946e13d
# ╠═ab24c367-0673-450b-aec1-0f68df6d906d
# ╠═40505c32-3d19-4c2c-95f5-a3df1432e643
# ╠═9d684cd0-7b6d-4518-9edc-9a7f42d9e408
# ╠═7b9fab16-4aef-4f81-8edd-1b08a09a904d
# ╠═b7e9b315-115b-414e-97a7-c843df36ed94
# ╠═cf1fe910-55fe-43ae-9368-11a9d6ab6efa
# ╠═a71a8ece-0a63-4129-bdf7-ef0add2ef81c
# ╠═8393f758-e0c8-4af4-88b3-c360afe7d363
# ╠═43c52fbc-2638-48f3-a96a-691881f8576c
# ╠═cfb8edbd-8739-4a5c-acde-5b04432c653e
# ╠═7eeac53f-8b55-4e39-ab27-fdc3ee13d1a0
# ╠═b0d286c6-9203-44fd-865e-f957df8d5b2c
# ╠═0466dc26-b7ad-4e33-b892-cda11b3c2a3f
# ╠═08da34b6-cdda-478e-8070-5f1a103b8866
# ╠═2c86c151-729f-471f-bba8-3a07f9e98cb5
# ╠═4837cd91-cdec-44ba-b59b-301946aa0874
# ╠═a5018eca-52c8-408c-8ff1-caf5cef45945
# ╠═466d1699-c75b-443f-b823-596c6c8362b9
# ╠═886243c7-e507-47f1-882d-7da022cab579
# ╠═c5dd42ef-613e-4308-a505-125d58e6233f
# ╠═dbcace88-ee69-4fab-a6c7-a6ae9c1c1a45
# ╠═8678bccb-484f-4876-bfa7-418bee26f0b6
# ╠═840131bd-0a88-41da-a8d2-0dcc38bfd8e5
# ╠═14b395e7-f6c5-4dff-bc5e-d7c6b20688e4
# ╠═2280d962-58fc-44e1-9e77-1a4428611204
# ╠═86d679f0-aa36-456c-af99-c8cc150200b4
# ╠═8a9202d2-6650-4ed2-9602-deecb91196d7
# ╠═02c7ff3e-babe-41b2-acba-eae2393141ab
# ╠═8ba29a81-385e-43e7-9f7e-0b709159e786
# ╠═2459937e-a35e-4f81-ab16-f191ee0963e0
# ╠═eca020dc-712e-4dbd-b14d-a156ed87c9d5
# ╠═f93b239b-552e-4371-87de-b63245b4a001
# ╠═1fae2aab-0f7b-4071-928a-317c6fa2d425
# ╠═c7bd78bf-70a4-4bb7-a664-e646c6dbf8c8
# ╠═c9e3fb05-bc3e-42c9-9331-88bfae4b9d59
# ╠═651c2d51-953b-4de8-802b-2c8affb79bf0
