FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# This Python workspace does not use Yarn. The inherited Yarn apt source has
# a stale signing key that prevents apt updates in Docker/GitHub CLI features.
# Remove that unused source; keep signature verification for all other sources.
RUN rm -f /etc/apt/sources.list.d/yarn.list
