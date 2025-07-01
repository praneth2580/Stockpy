# nextjs-frontend/Dockerfile
# Stage 1: Build the Next.js application
FROM node:18-alpine as builder

WORKDIR /app

# Copy package.json and package-lock.json and install dependencies
COPY package.json package-lock.json ./
RUN npm install

# Copy the rest of the application code (including your 'app' directory)
COPY . .

# Build the Next.js application
# This is where the actual build process happens, generating the .next folder.
RUN npm run build

# Stage 2: Serve the Next.js application
FROM node:18-alpine as runner

WORKDIR /app

ENV NODE_ENV production

# Copy only the necessary files for production from the builder stage.
# These files are the output of the 'npm run build' from the builder stage.
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/next.config.ts ./next.config.ts

# Expose the port Next.js listens on
EXPOSE 3000

# Command to start the Next.js production server.
# This command serves the already built application from the .next folder.
CMD ["npm", "start"]