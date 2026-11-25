set -eu
# RunPod injects its own per-pod CLI credential. ModelLab sends only an SSH public key.
mkdir -p /root/.ssh /run/sshd /workspace/modellab
chmod 700 /root/.ssh
printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
ssh-keygen -A
if ! command -v runpodctl >/dev/null && [ -z "${RUNPOD_API_KEY:-}" ]; then
  exit 11
fi
# sshd starts noninteractive sessions with a clean environment. Pass only the
# machine identity and CUDA paths needed by the worker into its interpreter.
{
  printf '#!/bin/bash\n'
  for var in PATH LD_LIBRARY_PATH CUDA_HOME RUNPOD_POD_ID NVIDIA_VISIBLE_DEVICES NVIDIA_DRIVER_CAPABILITIES; do
    if [ -n "${!var:-}" ]; then printf 'export %s=%q\n' "$var" "${!var}"; fi
  done
  printf 'exec %q "$@"\n' "$(command -v python3)"
} > /run/modellab-python
chmod 700 /run/modellab-python
guard_deadline="$MODELLAB_DEADLINE"
# An explicitly restarted recovery pod gets five minutes for artifact retrieval.
if [ -f /workspace/modellab/config.json ] && [ "$(date +%s)" -ge "$guard_deadline" ]; then
  guard_deadline=$(( $(date +%s) + 300 ))
fi
(
  while [ "$(date +%s)" -lt "$guard_deadline" ]; do sleep 3; done
  while :; do
    if ! (runpodctl pod stop "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID") >/dev/null 2>&1; then
      if [ -n "${RUNPOD_API_KEY:-}" ]; then
        curl --silent --fail --max-time 10 -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
          -H 'Content-Type: application/json' -d '{"action":"stop"}' \
          "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID/action" >/dev/null || true
      fi
    fi
    sleep 5
  done
) &
echo "$!" > /run/modellab-guard.pid
exec /usr/sbin/sshd -D -e -o PermitRootLogin=prohibit-password -o PasswordAuthentication=no -o AllowTcpForwarding=no -o X11Forwarding=no
