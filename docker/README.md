# THREATCAST Dockerfiles

Build contexts assume the repository root as build context:

```powershell
docker build -f docker/backend.Dockerfile -t threatcast-backend .
docker build -f docker/frontend.Dockerfile -t threatcast-frontend .
docker build -f docker/ml.Dockerfile -t threatcast-ml .
```
