# =============================================================================
# AcousticSpace Frontend — Production Dockerfile
# Multi-stage build: compile the Vite/React app, then serve the static
# output with a lightweight nginx — no Node.js process runs in production.
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — Build
# ---------------------------------------------------------------------------
FROM node:20-alpine AS build

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

# VITE_API_BASE_URL defaults to "/api" in the app code, which is what the
# reverse-proxy (nginx) expects, so no build-arg is required for the
# standard docker-compose deployment. Override at build time if needed:
#   docker build --build-arg VITE_API_BASE_URL=https://api.example.com .
ARG VITE_API_BASE_URL
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2 — Serve
# ---------------------------------------------------------------------------
FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html

# Client-side routing (React Router) — always fall back to index.html
RUN printf 'server {\n\
    listen 80;\n\
    root /usr/share/nginx/html;\n\
    index index.html;\n\
    location / {\n\
        try_files $uri /index.html;\n\
    }\n\
}\n' > /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]