import candidates from "@/data/candidates.json";

export const runtime = "nodejs";

export function GET() {
  return Response.json(candidates, {
    headers: { "Cache-Control": "public, max-age=300, s-maxage=300" },
  });
}
