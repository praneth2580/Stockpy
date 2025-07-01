// nextjs-frontend/src/app/api/predict/route.js
import { NextResponse } from "next/server";

// ML_SERVICE_BASE_URL is loaded from .env.local during local dev,
// and from Render's environment variables in production.
const ML_SERVICE_BASE_URL = process.env.ML_SERVICE_BASE_URL;

export async function POST(request) {
  if (!ML_SERVICE_BASE_URL) {
    console.error("ML_SERVICE_BASE_URL is not defined!");
    return NextResponse.json(
      { detail: "Server configuration error: ML service URL not set." },
      { status: 500 }
    );
  }

  try {
    const requestBody = await request.json();
    console.log("Forwarding request to ML service:", requestBody);

    const mlServiceResponse = await fetch(`${ML_SERVICE_BASE_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    if (!mlServiceResponse.ok) {
      const errorData = await mlServiceResponse.json();
      console.error("ML Service Error:", errorData);
      return NextResponse.json(
        {
          detail:
            errorData.detail || "ML Service failed to return a valid response.",
        },
        { status: mlServiceResponse.status }
      );
    }

    const mlServiceData = await mlServiceResponse.json();
    return NextResponse.json(mlServiceData);
  } catch (error) {
    console.error("API Route Error:", error);
    return NextResponse.json(
      { detail: "Internal Server Error: " + error.message },
      { status: 500 }
    );
  }
}
