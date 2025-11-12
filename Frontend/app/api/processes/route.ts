import { NextRequest, NextResponse } from 'next/server';
import { getProcessesPayload } from '../../../lib/processData';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = request.nextUrl;
    const algorithm = searchParams.get('algorithm') ?? undefined;
    const payload = await getProcessesPayload(algorithm);

    return NextResponse.json(payload, {
      headers: {
        'Cache-Control': 'no-store, must-revalidate'
      }
    });
  } catch (error) {
    console.error('Failed to load process data', error);
    return NextResponse.json({ error: 'Failed to load process data' }, { status: 500 });
  }
}
