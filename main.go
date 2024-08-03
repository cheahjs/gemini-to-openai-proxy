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
	"strings"
	"sync/atomic"
	"time"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

const (
	openAIEmbeddingsEndpoint = "/v1/embeddings"
	openAIModelsEndpoints    = "/v1/models"
)

var (
	GeminiApiKey  = os.Getenv("GEMINI_API_KEY")
	GeminiApiKeys = strings.Split(GeminiApiKey, ";")
	ListenAddr    = os.Getenv("LISTEN_ADDR")
	MetricsAddr   = os.Getenv("METRICS_ADDR")
	geminiClients []*genai.Client
	currentClient atomic.Int32

	requestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "requests_total",
			Help: "Total number of requests",
		},
		[]string{"path", "method", "status"},
	)
	embeddingBatchSize = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "embedding_batch_size",
			Help:    "Size of embedding batches",
			Buckets: prometheus.LinearBuckets(1, 1, 10),
		},
		[]string{"model"},
	)
	requestLatency = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "request_latency_seconds",
			Help:    "Request latency in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"path", "method"},
	)
)

func init() {
	prometheus.MustRegister(requestsTotal)
	prometheus.MustRegister(embeddingBatchSize)
	prometheus.MustRegister(requestLatency)
}

func embeddingsHandler(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
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
		requestsTotal.WithLabelValues(r.URL.Path, r.Method, http.StatusText(http.StatusMethodNotAllowed)).Inc()
		requestLatency.WithLabelValues(r.URL.Path, r.Method).Observe(time.Since(start).Seconds())
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
		requestsTotal.WithLabelValues(r.URL.Path, r.Method, http.StatusText(http.StatusBadRequest)).Inc()
		requestLatency.WithLabelValues(r.URL.Path, r.Method).Observe(time.Since(start).Seconds())
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
		requestsTotal.WithLabelValues(r.URL.Path, r.Method, http.StatusText(http.StatusBadRequest)).Inc()
		requestLatency.WithLabelValues(r.URL.Path, r.Method).Observe(time.Since(start).Seconds())
		return
	}

	useIndex := currentClient.Add(1) % int32(len(geminiClients))
	requestLogger.Info().Str("model", openAIReq.Model).Int32("client", useIndex).Msg("Processing request")

	embeddingModel := geminiClients[useIndex].EmbeddingModel(openAIReq.Model)

	geminiBatchReq, err := openai.ConvertOpenAIRequestToGemini(&openAIReq, embeddingModel)
	if err != nil {
		http.Error(w, "Bad Request", http.StatusBadRequest)
		requestLogger.
			Error().
			Err(errors.Wrap(err, "failed to convert OpenAI request to Gemini request")).
			Int("status-code", http.StatusBadRequest).
			Msg("")
		requestsTotal.WithLabelValues(r.URL.Path, r.Method, http.StatusText(http.StatusBadRequest)).Inc()
		requestLatency.WithLabelValues(r.URL.Path, r.Method).Observe(time.Since(start).Seconds())
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
		requestsTotal.WithLabelValues(r.URL.Path, r.Method, http.StatusText(http.StatusInternalServerError)).Inc()
		requestLatency.WithLabelValues(r.URL.Path, r.Method).Observe(time.Since(start).Seconds())
		return
	}

	openAIResp := openai.ConvertGeminiResponseToOpenAI(geminiBatchResp, openAIReq.Model)

	embeddingBatchSize.WithLabelValues(openAIReq.Model).Observe(float64(len(openAIReq.Input.([]interface{}))))

	w.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w).Encode(openAIResp)
	if err != nil {
		requestLogger.
			Error().
			Err(errors.Wrap(err, "failed to encode response")).
			Int("status-code", http.StatusInternalServerError).
			Msg("")
		requestsTotal.WithLabelValues(r.URL.Path, r.Method, http.StatusText(http.StatusInternalServerError)).Inc()
		requestLatency.WithLabelValues(r.URL.Path, r.Method).Observe(time.Since(start).Seconds())
		return
	}

	requestsTotal.WithLabelValues(r.URL.Path, r.Method, http.StatusText(http.StatusOK)).Inc()
	requestLatency.WithLabelValues(r.URL.Path, r.Method).Observe(time.Since(start).Seconds())
}

func modelsHandler(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
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
		requestsTotal.WithLabelValues(r.URL.Path, r.Method, http.StatusText(http.StatusMethodNotAllowed)).Inc()
		requestLatency.WithLabelValues(r.URL.Path, r.Method).Observe(time.Since(start).Seconds())
		return
	}

	var models []*openai.ModelResponseData

	iter := geminiClients[0].ListModels(r.Context())
	for {
		m, err := iter.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			http.Error(w, "Internal Server Error", http.StatusInternalServerError)
			requestLogger.Error().Err(err).Msg("Failed to list models")
			requestsTotal.WithLabelValues(r.URL.Path, r.Method, http.StatusText(http.StatusInternalServerError)).Inc()
			requestLatency.WithLabelValues(r.URL.Path, r.Method).Observe(time.Since(start).Seconds())
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
		requestsTotal.WithLabelValues(r.URL.Path, r.Method, http.StatusText(http.StatusInternalServerError)).Inc()
		requestLatency.WithLabelValues(r.URL.Path, r.Method).Observe(time.Since(start).Seconds())
		return
	}

	requestsTotal.WithLabelValues(r.URL.Path, r.Method, http.StatusText(http.StatusOK)).Inc()
	requestLatency.WithLabelValues(r.URL.Path, r.Method).Observe(time.Since(start).Seconds())
}

func main() {
	if ListenAddr == "" {
		ListenAddr = ":8080"
	}
	if GeminiApiKey == "" {
		log.Fatal().Msg("GEMINI_API_KEY is required")
	}
	currentClient.Store(0)
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnixMs
	for _, key := range GeminiApiKeys {
		client, err := genai.NewClient(context.Background(), option.WithAPIKey(key))
		if err != nil {
			log.
				Fatal().
				Err(errors.Wrap(err, "failed to create Gemini client")).
				Int("status-code", http.StatusInternalServerError).
				Msg("")
			return
		}
		geminiClients = append(geminiClients, client)
	}
	http.HandleFunc(openAIEmbeddingsEndpoint, embeddingsHandler)
	http.HandleFunc(openAIModelsEndpoints, modelsHandler)

	if MetricsAddr != "" {
		go func() {
			mux := http.NewServeMux()
			mux.Handle("/metrics", promhttp.Handler())
			log.Info().Msgf("Exposing metrics on %s/metrics", MetricsAddr)
			log.Fatal().Err(http.ListenAndServe(MetricsAddr, mux)).Msg("Failed to listen and serve metrics")
		}()
	}

	log.Info().Msgf("Listening on %s", ListenAddr)
	log.Fatal().Err(http.ListenAndServe(ListenAddr, nil)).Msg("Failed to listen and serve")
}
