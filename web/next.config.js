/** @type {import('next').NextConfig} */
const isStaticExport = process.env.STATIC_EXPORT === '1'

const nextConfig = {
  reactStrictMode: true,
  ...(isStaticExport && {
    output: 'export',
    trailingSlash: true,
    basePath: process.env.NEXT_PUBLIC_BASE_PATH || '',
    images: { unoptimized: true },
  }),
}

module.exports = nextConfig
