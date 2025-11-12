import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Scheduling Algorithm Simulator',
  description: 'Real-time CPU scheduling visualizer driven by live dataset snapshots.'
};

export default function RootLayout({
  children
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
