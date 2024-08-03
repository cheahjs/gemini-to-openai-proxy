# gemini-to-openai-proxy
Call Gemini (https://ai.google.dev) embedding models with OpenAI-compatible endpoints

## Deployment

### Using `docker run`

To deploy using `docker run`, you can use the following command:

```sh
docker run -d -p 8080:8080 -e GEMINI_API_KEY=<your-gemini-api-key> ghcr.io/cheahjs/gemini-to-openai-proxy:latest
```

Replace `<your-gemini-api-key>` with your actual Gemini API key.

### Using `docker compose`

To deploy using `docker compose`, you can use the provided `docker-compose.yaml` file. First, create a `.env` file with the following content:

```sh
GEMINI_API_KEY=<your-gemini-api-key>
LISTEN_ADDR=:8080
```

Then, run the following command:

```sh
docker-compose up -d
```

### Using `go install`

To deploy using `go install`, you need to have Go installed on your machine. Run the following commands:

```sh
go install github.com/cheahjs/gemini-to-openai-proxy@latest
GEMINI_API_KEY=<your-gemini-api-key> LISTEN_ADDR=:8080 gemini-to-openai-proxy
```

Replace `<your-gemini-api-key>` with your actual Gemini API key.
