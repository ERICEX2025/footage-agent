#!/usr/bin/env bash
# Deploy JARVIS to the team's K8s namespace behind Ingress path /jarvis.
# (/app belongs to the team's main demo app - don't touch it.)
# Run inside the event VM:  ./deploy.sh
set -euo pipefail
cd "$(dirname "$0")"

TEAM_CONFIG=$(ls /config/*.config | head -1)
cfg() { grep "^$1=" "$TEAM_CONFIG" | cut -d= -f2-; }
USERNAME=$(cfg USERNAME)
NS="$USERNAME"
APP_HOST="video-lab-team-${USERNAME#team-}.cosmos.vastdata.com"
APP=jarvis
export KUBECONFIG=/config/${NS}-k8s.yaml

kubectl -n "$NS" create configmap "$APP-code" \
  --from-file=app.py --from-file=agent.py --from-file=vss_client.py \
  --from-file=voice.py --from-file=requirements.txt \
  --dry-run=client -o yaml | kubectl apply -f -

# Pods cannot resolve the public VSS hostname; use the in-cluster backend service.
kubectl -n "$NS" create secret generic "$APP-env" \
  --from-literal=INGRESS_URL="http://video-backend-service.$NS.svc:8000" \
  --from-literal=USERNAME="$USERNAME" \
  --from-literal=PASSWORD="$(cfg PASSWORD)" \
  --from-literal=GPU_BEARER_TOKEN="$(cfg GPU_BEARER_TOKEN)" \
  --from-literal=WANDB_API_KEY="${WANDB_API_KEY:-$(cfg WANDB_API_KEY)}" \
  --from-literal=WANDB_TEAM="${WANDB_TEAM:-}" \
  --from-literal=WANDB_PROJECT="${WANDB_PROJECT:-}" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl -n "$NS" apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata: {name: $APP, labels: {app: $APP}}
spec:
  replicas: 1
  selector: {matchLabels: {app: $APP}}
  template:
    metadata: {labels: {app: $APP}}
    spec:
      containers:
      - name: app
        image: python:3.12-slim
        ports: [{containerPort: 8080}]
        envFrom: [{secretRef: {name: $APP-env}}]
        volumeMounts: [{name: code, mountPath: /code}]
        workingDir: /code
        command: ["bash", "-c"]
        args:
        - |
          set -e
          pip install --no-cache-dir -q -r requirements.txt
          exec streamlit run app.py --server.port 8080 --server.address 0.0.0.0 \
            --server.headless true --server.enableCORS false --server.enableXsrfProtection false \
            --browser.gatherUsageStats false
        readinessProbe:
          httpGet: {path: /_stcore/health, port: 8080}
          initialDelaySeconds: 20
          periodSeconds: 5
          failureThreshold: 60
      volumes:
      - name: code
        configMap: {name: $APP-code}
---
apiVersion: v1
kind: Service
metadata: {name: $APP, labels: {app: $APP}}
spec:
  selector: {app: $APP}
  ports: [{name: http, port: 80, targetPort: 8080}]
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: $APP
  labels: {app: $APP}
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /\$2
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-body-size: "20m"
spec:
  ingressClassName: nginx
  rules:
  - host: $APP_HOST
    http:
      paths:
      - path: /jarvis(/|$)(.*)
        pathType: ImplementationSpecific
        backend: {service: {name: $APP, port: {number: 80}}}
EOF

kubectl -n "$NS" rollout restart deploy/$APP
kubectl -n "$NS" rollout status deploy/$APP --timeout=300s
curl -sS -o /dev/null -w "jarvis HTTP %{http_code}\n" "http://$APP_HOST/jarvis/"
