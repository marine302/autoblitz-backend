// src/types/index.ts
export interface CoinRecommendation {
  symbol: string;
  exchange: string;
  current_price: number;
  recommendation_score: number;
  grade: 'A+' | 'A' | 'B+' | 'B' | 'C+' | 'C' | 'D';
  volatility_24h: number;
  volume_24h: number;
  market_cap: number;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
  pump_dump_risk: number;
  last_updated: string;
}

export interface MarketSummary {
  total_coins: number;
  active_exchanges: number;
  high_grade_count: number;
  medium_grade_count: number;
  low_grade_count: number;
  avg_score: number;
  market_trend: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
}

export interface ApiResponse<T> {
  data: T;
  message: string;
  status: 'success' | 'error';
}

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
