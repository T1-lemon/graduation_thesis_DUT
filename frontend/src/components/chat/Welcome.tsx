import React, { useEffect, useState } from 'react'
import styled from 'styled-components'

import { useAppSelector } from '../../redux/hooks'

const Robot = require('../../assets/img/robot.gif')

type Props = {}

export default function Welcome({}: Props) {
  const { currentUser } = useAppSelector((state) => state.auths)

  return (
    <Container>
      <img src={Robot} alt='' />
      <h1>
        Welcome,{' '}
        <span>
          {' '}
          {currentUser.firstName} {currentUser.lastName}
        </span>
      </h1>
      <h3>Please select a chat to Start messaging.</h3>
    </Container>
  )
}

const Container = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  color: #5a636c;
  flex-direction: column;
  background-color: #ebf7f6;
  img {
    height: 20rem;
  }
  span {
    color: #1fc29d;
  }
`
