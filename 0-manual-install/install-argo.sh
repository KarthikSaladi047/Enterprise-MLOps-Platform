# infra runs all required platform software
kubectl label nodes <infra-node-name>   node-role.kubernetes.io/infra=true

# apps runs UI, API, model consumers
kubectl label nodes <app-node-name>     node-role.kubernetes.io/apps=true
kubectl taint  nodes <app-node-name>    dedicated=apps:NoSchedule

# compute runs Kubeflow pipelines
kubectl label nodes <compute-node-name> node-role.kubernetes.io/compute=true
kubectl taint  nodes <compute-node-name> dedicated=compute:NoSchedule

# install argocd
helm install argocd argo/argo-cd \
  --namespace argocd \
  --create-namespace \
  --set server.service.type=NodePort

# kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d; echo