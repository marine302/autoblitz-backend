// src/services/api.ts
import axios from 'axios';
import { CoinRecommendation, MarketSummary, ApiResponse } from '../types';

const API_BASE_URL = 'http://localhost:8000/api/v1';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = {
  // 추천 코인 목록 조회
  async getRecommendations(limit: number = 20): Promise<CoinRecommendation[]> {
    const response = await apiClient.get<ApiResponse<CoinRecommendation[]>>(
      `/recommendations/?limit=${limit}`
    );
    return response.data.data;
  },

  // 시장 요약 정보
  async getMarketSummary(): Promise<MarketSummary> {
    const response = await apiClient.get<ApiResponse<MarketSummary>>(
      '/recommendations/summary'
    );
    return response.data.data;
  },

  // 특정 코인 상세 정보
  async getCoinDetail(symbol: string): Promise<CoinRecommendation> {
    const response = await apiClient.get<ApiResponse<CoinRecommendation>>(
      `/recommendations/${symbol}`
    );
    return response.data.data;
  },

  // 헬스 체크
  async healthCheck(): Promise<any> {
    const response = await apiClient.get('/health');
    return response.data;
  },
};
