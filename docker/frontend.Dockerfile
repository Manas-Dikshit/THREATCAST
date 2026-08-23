FROM node:20-alpine

WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install || npm install --no-package-lock

COPY frontend/ ./
EXPOSE 5173
CMD ["npm", "run", "dev"]
