package main

import (
	"context"
	"encoding/json"
	"github.com/cheahjs/gemini-to-openai-proxy/pkg/openai"
	"github.com/google/generative-ai-go/genai"
	"github.com/pkg/errors"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
	"io"
	"net/http"
	"os"
	"slices"
)

const (
	openAIEmbeddingsEndpoint = "/v1/embeddings"
	openAIModelsEndpoints    = "/v1/models"
)

var (
	GeminiApiKey = os.Getenv("GEMINI_API_KEY")
	ListenAddr   = os.Getenv("LISTEN_ADDR")
	geminiClient *genai.Client
)

func embeddingsHandler(w http.ResponseWriter, r *http.Request) {
	requestLogger := log.With().
		Str("path", r.URL.Path).
		Str("user-agent", r.Header.Get("User-Agent")).
		Logger()

	if r.Method != http.MethodPost {
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		requestLogger.
			Error().
			Int("status-code", http.StatusMethodNotAllowed).
			Msg("")
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Bad Request", http.StatusBadRequest)
		requestLogger.
			Error().
			Err(errors.Wrap(err, "failed to read request body")).
			Int("status-code", http.StatusBadRequest).
			Msg("")
		return
	}

	var openAIReq openai.EmbedRequest
	err = json.Unmarshal(body, &openAIReq)
	if err != nil {
		http.Error(w, "Bad Request", http.StatusBadRequest)
		requestLogger.
			Error().
			Err(errors.Wrap(err, "failed to unmarshal request body")).
			Int("status-code", http.StatusBadRequest).
			Msg("")
		return
	}
	requestLogger.Info().Str("model", openAIReq.Model).Msg("Processing request")

	embeddingModel := geminiClient.EmbeddingModel(openAIReq.Model)

	geminiBatchReq, err := openai.ConvertOpenAIRequestToGemini(&openAIReq, embeddingModel)
	if err != nil {
		http.Error(w, "Bad Request", http.StatusBadRequest)
		requestLogger.
			Error().
			Err(errors.Wrap(err, "failed to convert OpenAI request to Gemini request")).
			Int("status-code", http.StatusBadRequest).
			Msg("")
		return
	}

	geminiBatchResp, err := embeddingModel.BatchEmbedContents(r.Context(), geminiBatchReq)
	if err != nil {
		http.Error(w, "Internal Server Error", http.StatusInternalServerError)
		requestLogger.
			Error().
			Err(errors.Wrap(err, "failed to batch embed contents")).
			Int("status-code", http.StatusInternalServerError).
			Msg("")
		return
	}

	openAIResp := openai.ConvertGeminiResponseToOpenAI(geminiBatchResp, openAIReq.Model)

	w.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w).Encode(openAIResp)
	if err != nil {
		requestLogger.
			Error().
			Err(errors.Wrap(err, "failed to encode response")).
			Int("status-code", http.StatusInternalServerError).
			Msg("")
		return
	}
}

func modelsHandler(w http.ResponseWriter, r *http.Request) {
	requestLogger := log.With().
		Str("path", r.URL.Path).
		Str("user-agent", r.Header.Get("User-Agent")).
		Logger()

	if r.Method != http.MethodGet {
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		requestLogger.
			Error().
			Int("status-code", http.StatusMethodNotAllowed).
			Msg("")
		return
	}

	var models []*openai.ModelResponseData

	iter := geminiClient.ListModels(r.Context())
	for {
		m, err := iter.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			http.Error(w, "Internal Server Error", http.StatusInternalServerError)
			requestLogger.Error().Err(err).Msg("Failed to list models")
			return
		}
		if !slices.Contains(m.SupportedGenerationMethods, "embedContent") {
			continue
		}
		models = append(models, &openai.ModelResponseData{
			Object:  "model",
			ID:      m.Name,
			Created: 0,
			OwnedBy: "google",
		})
	}

	err := json.NewEncoder(w).Encode(&openai.ModelResponse{
		Object: "list",
		Data:   models,
	})
	if err != nil {
		requestLogger.
			Error().
			Err(errors.Wrap(err, "failed to encode response")).
			Int("status-code", http.StatusInternalServerError).
			Msg("")
		return
	}
}

func main() {
	if ListenAddr == "" {
		ListenAddr = ":8080"
	}
	if GeminiApiKey == "" {
		log.Fatal().Msg("GEMINI_API_KEY is required")
	}
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnixMs
	var err error
	geminiClient, err = genai.NewClient(context.Background(), option.WithAPIKey(GeminiApiKey))
	if err != nil {
		log.
			Fatal().
			Err(errors.Wrap(err, "failed to create Gemini client")).
			Int("status-code", http.StatusInternalServerError).
			Msg("")
		return
	}
	defer func(geminiClient *genai.Client) {
		err := geminiClient.Close()
		if err != nil {
			log.Error().Err(err).Msg("Failed to close Gemini client")
		}
	}(geminiClient)
	http.HandleFunc(openAIEmbeddingsEndpoint, embeddingsHandler)
	http.HandleFunc(openAIModelsEndpoints, modelsHandler)
	log.Info().Msgf("Listening on %s", ListenAddr)
	log.Fatal().Err(http.ListenAndServe(ListenAddr, nil)).Msg("Failed to listen and serve")
}
