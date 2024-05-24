FROM golang:1.22 as build
WORKDIR /go/src/app
ADD . /go/src/app

RUN go get -d -v ./...
RUN go build -o /go/bin/app github.com/cheahjs/gemini-to-openai-proxy

FROM gcr.io/distroless/static-debian11:debug
COPY --from=build /go/bin/app /
ENTRYPOINT ["/app"]
